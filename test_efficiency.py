import gc
import numpy as np
import time
import torch
from timm.models import create_model

import modeling_finetune
from other_models.MVD import modeling_finetune
from other_models.InternVideo2_single_modality import models


def main(model_type, with_flash=False):
    gc.collect()
    torch.cuda.empty_cache()

    steps = 1000
    x_ = np.random.randn(1, 3, 16, 224, 224).astype(np.float32)
    x2_ = np.random.randn(1, 3, 8, 224, 224).astype(np.float32)

    device = torch.device("cuda")

    if model_type == "VideoMAE-S":
        # VideoMAE ViT-S
        model = create_model(
                "vit_small_patch16_224",
                pretrained=False,
                num_classes=2,
                all_frames=16,
                tubelet_size=2,
                fc_drop_rate=0.0,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                drop_block_rate=None,
                use_checkpoint=False,
                final_reduction="fc_norm",
                init_scale=0.001,
                use_flash_attn=with_flash
            )
    if model_type == "VideoMAE-B":
        # VideoMAE ViT-S
        model = create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=2,
                all_frames=16,
                tubelet_size=2,
                fc_drop_rate=0.0,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                drop_block_rate=None,
                use_checkpoint=False,
                final_reduction="fc_norm",
                init_scale=0.001,
                use_flash_attn=with_flash
            )
    elif model_type == "MVD-S":
        # VideoMAE ViT-S
        model = create_model(
                "mvd_vit_small_patch16_224",
                pretrained=False,
                num_classes=2,
                all_frames=16,
                tubelet_size=2,
                fc_drop_rate=0.0,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                drop_block_rate=None,
                use_checkpoint=False,
                final_reduction="fc_norm",
                use_cls_token=False,
                init_scale=0.001,
                use_flash_attn=with_flash
            )
    elif model_type == "MVD-B":
        # VideoMAE ViT-S
        model = create_model(
                "mvd_vit_base_patch16_224",
                pretrained=False,
                num_classes=2,
                all_frames=16,
                tubelet_size=2,
                fc_drop_rate=0.0,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                drop_block_rate=None,
                use_checkpoint=False,
                final_reduction="fc_norm",
                use_cls_token=False,
                init_scale=0.001,
                use_flash_attn=with_flash
            )
    elif model_type == "ViViT-B":
        # VideoMAE ViT-S
        model = create_model(
                "vit_base_patch16_224",
                pretrained=False,
                num_classes=2,
                all_frames=16,
                tubelet_size=2,
                fc_drop_rate=0.0,
                drop_rate=0.0,
                drop_path_rate=0.1,
                attn_drop_rate=0.0,
                drop_block_rate=None,
                use_checkpoint=False,
                final_reduction="fc_norm",
                init_scale=0.001,
                use_flash_attn=with_flash
            )
    elif model_type == "InternVideo2-S":
        model = create_model(
        "internvideo2_small_patch14_224",
        pretrained=False,
        num_classes=2,
        num_frames=8,
        tubelet_size=1,
        sep_pos_embed=False,
        fc_drop_rate=0.0,
        drop_path_rate=0.1,
        head_drop_path_rate=0.1,
        use_checkpoint=False,
        checkpoint_num=0,
        init_scale=0.001,
        init_values=1e-5,
        layerscale_no_force_fp32=False,
        qkv_bias=False,
        use_flash_attn=with_flash,
        use_fused_rmsnorm=with_flash,
        use_fused_mlp=with_flash,
    )
    elif model_type == "InternVideo2-B":
        model = create_model(
        "internvideo2_base_patch14_224",
        pretrained=False,
        num_classes=2,
        num_frames=8,
        tubelet_size=1,
        sep_pos_embed=False,
        fc_drop_rate=0.0,
        drop_path_rate=0.1,
        head_drop_path_rate=0.1,
        use_checkpoint=False,
        checkpoint_num=0,
        init_scale=0.001,
        init_values=1e-5,
        layerscale_no_force_fp32=False,
        qkv_bias=False,
        use_flash_attn=with_flash,
        use_fused_rmsnorm=with_flash,
        use_fused_mlp=with_flash,
    )
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params:', n_parameters)

    if model_type in ("InternVideo2-S", "InternVideo2-B"):
        x = torch.tensor(x2_).to(device)
    else:
        x = torch.tensor(x_).to(device)

    model.to(device)
    
    model.eval()

    forward_times = []
    memory_usage = []

    #print(f"[{model_type}] gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} memory\n")

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            out = model(x)
        for i in range(steps):
            start = time.time()
            with torch.cuda.amp.autocast():
                out = model(x)
            end = time.time()
            t = end - start
            memuse = torch.cuda.max_memory_allocated(device=None) / (1024**2)
            memory_usage.append(memuse)
            #print(f'step {t}: {t} s')
            #print(f"\t[{model_type}] gpu used {torch.cuda.max_memory_allocated(device=None) / (1024**2)} memory")
            torch.cuda.reset_peak_memory_stats(device=None)
            forward_times.append(t)

        avg_time = sum(forward_times) / steps
        avg_mem = sum(memory_usage) / steps
        print(f'{model_type} | Average time: {round(1000 * avg_time)} ms')
        print(f'{model_type} | Average FPS: {1 / avg_time:.2f}')
        print(f'{model_type} | Average GPU memory use: {avg_mem:.2f}')
        del x 
        del model


if __name__ == "__main__":
    with_flash = True
    main(model_type="VideoMAE-S", with_flash=with_flash)
    print("##############################")
    main(model_type="MVD-S", with_flash=with_flash)
    print("##############################")
    main(model_type="MVD-B", with_flash=with_flash)
    print("##############################")
    main(model_type="ViViT-B", with_flash=with_flash)
    print("##############################")
    main(model_type="InternVideo2-S", with_flash=with_flash)
    print("##############################")
    main(model_type="InternVideo2-B", with_flash=with_flash)
    print("##############################")

