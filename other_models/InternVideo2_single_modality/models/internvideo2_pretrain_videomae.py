import math
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn

import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange

from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from .flash_attention_class import FlashAttention
from .internvideo2 import Block, AttentionPoolingBlock, RMSNorm
from .internvideo2 import PatchEmbed_VideoMAE
from flash_attn.modules.mlp import FusedMLP
from flash_attn.ops.rms_norm import DropoutAddRMSNorm

from modeling_pretrain import PretrainVisionTransformerDecoder, PatchEmbed, get_sinusoid_encoding_table


class iv2_PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(
            # VideoMAE specific
            self, 
            num_classes=0, 
            qk_scale=None, 
            drop_rate=0., 
            attn_drop_rate=0.,
            norm_layer=nn.LayerNorm,
            # IV2
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.25,
            embed_dim: int = 1408,
            num_heads: int = 16,
            mlp_ratio: float = 4.3637,
            init_values: float = 1e-5,
            qk_normalization: bool = True,
            depth: int = 40,
            use_flash_attn: bool = True,
            use_fused_rmsnorm: bool = True,
            use_fused_mlp: bool = True,
            fused_mlp_heuristic: int = 1,
            layerscale_no_force_fp32: bool = False,
            num_frames: int = 8,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            use_checkpoint: bool = False,
            checkpoint_num: int = 0,):
        super().__init__()

        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, print(
            'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent')
        print(mlp_ratio)

        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim

        if use_fused_rmsnorm:
            norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = PatchEmbed_VideoMAE(
            img_size, patch_size, in_chans, embed_dim,
            num_frames=num_frames, tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # stolen from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/models_vit.py#L65-L73C17
        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            print("Use seperable position embedding")
            grid_size = self.patch_embed.grid_size
            self.grid_size = grid_size
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, grid_size[1] * grid_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, grid_size[0], embed_dim))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            print("Use joint position embedding")
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        print(f"Droppath rate: {dpr}")
        print(f"Checkpoint list: {with_cp_list}")

        # From VideoMAE ViT
        self.num_classes = num_classes
        self.num_features = self.embed_dim  # num_features for consistency with other models
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn, use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  use_fused_rmsnorm=use_fused_rmsnorm)
            for i in range(depth)])
        # We don't need aggregation when we try to reconstruct masked tokens
        # self.clip_projector = AttentionPoolingBlock(
        #     dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
        #     drop=0., attn_drop=0., norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim)

        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.init_pos_embed()
        #trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def init_pos_embed(self):
        print("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            # trunc_normal_(self.pos_embed_spatial, std=.02)
            # trunc_normal_(self.pos_embed_temporal, std=.02)
            # trunc_normal_(self.pos_embed_cls, std=.02)
            pos_embed_spatial = get_2d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1], 
                self.patch_embed.grid_size[1], # height & weight
            )
            self.pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_spatial).float().unsqueeze(0))
            pos_embed_temporal = get_1d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1], 
                self.patch_embed.grid_size[0], # t_size
            )
            self.pos_embed_temporal.data.copy_(torch.from_numpy(pos_embed_temporal).float().unsqueeze(0))
        else:
            # trunc_normal_(self.pos_embed, std=.02)
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1], 
                self.patch_embed.grid_size[1], # height & weight
                self.patch_embed.grid_size[0], # t_size
                cls_token=False
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        _, _, T, _, _ = x.shape
        x = self.patch_embed(x)

        # add pos_embed
        if self.sep_pos_embed:
            #print("form pos embed")
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.grid_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.grid_size[1] * self.grid_size[2],
                dim=1,
            )
        else:
            #print("use existing pos embed")
            pos_embed = self.pos_embed

        x = x + pos_embed

        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible

        residual = None
        for blk in self.blocks:
            if isinstance(x_vis, tuple) and len(x_vis) == 2:
                x_vis, residual = x_vis
            x_vis = blk(x_vis, residual=residual)
        if isinstance(x_vis, tuple) and len(x_vis) == 2:
            x_vis, residual = x_vis
            if residual is not None:
                x_vis = x_vis + residual

        # print(f"After blocks, x_vis is: ", type(x_vis))
        # if isinstance(x_vis, tuple):
        #     print(len(x_vis), x_vis[0].shape, x_vis[1].shape)
        # else:
        #     print(x_vis.shape)
        
        #x_vis = self.clip_projector(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis

    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.head(x)
        return x
    

class PretrainVideoMAEInternVideo2(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=14, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=588, #  decoder_num_classes=768, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 layerscale_no_force_fp32: bool = False,
                 num_frames: int = 8,
                 tubelet_size: int = 1,
                 sep_pos_embed: bool = False,
                 checkpoint_num: int = 0,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 ):
        super().__init__()
        self.encoder = iv2_PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            #
            use_flash_attn=True,
            use_fused_rmsnorm=True,
            use_fused_mlp=True,
            fused_mlp_heuristic=1,
            layerscale_no_force_fp32=layerscale_no_force_fp32,
            num_frames=num_frames,
            sep_pos_embed=sep_pos_embed,
            checkpoint_num=checkpoint_num
            )

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            embed_dim=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint)

        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, decoder_embed_dim)

        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        _, _, T, _, _ = x.shape
        x_vis = self.encoder(x, mask) # [B, N_vis, C_e]
        x_vis = self.encoder_to_decoder(x_vis) # [B, N_vis, C_d]
        B, N, C = x_vis.shape
        # we don't unshuffle the correct visible token order, 
        # but shuffle the pos embedding accorddingly.
        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1) # [B, N, C_d]
        x = self.decoder(x_full, pos_emd_mask.shape[1]) # [B, N_mask, 3 * 16 * 16]

        return x
    

@register_model
def pretrain_videomae_internvideo2_patch14_224(pretrained=False, **kwargs):
    model = PretrainVideoMAEInternVideo2(
        img_size=224, patch_size=14, 
        encoder_embed_dim=384, encoder_depth=12, encoder_num_heads=6, encoder_num_classes=0, 
        decoder_embed_dim=192, decoder_num_heads=3, decoder_num_classes=588, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), mlp_ratio=4.0, # 48/11, 
        **kwargs
    )
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224

    # model = pretrain_internvideo2_1B_patch14_224(clip_return_layer=6).cuda().half()
    model = pretrain_internvideo2_6B_patch14_224(clip_return_layer=1).cuda().half()
    # print(model)

    # flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda().half())
    # s = time.time()
    # print(flop_count_table(flops, max_depth=1))
    # print(time.time()-s)

    mask = torch.cat([
        torch.zeros(1, 1),
        torch.ones(1, 8 * int(16 * 16 * 0.75)),
        torch.zeros(1, 8 * int(16 * 16 * 0.25)),
    ], dim=-1).to(torch.bool).cuda()
    
    output = model(torch.rand(4, 3, num_frames, img_size, img_size).cuda().half(), mask.repeat(4, 1))
    print(output[0].shape)
    print(output[1].shape)
