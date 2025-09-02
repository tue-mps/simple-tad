import os
import json
import torch
import torch.nn.functional as F
from transformers import VivitModel

# from huggingface_hub import list_models
#
# # List models containing "vivit"
# models = list_models(filter="vivit")
# for model in models:
#     print(model.modelId)
# exit(0)

def adapt_vivit_pos_embed(checkpoint, num_temp_vivit=16, num_temp_vidmae=8, img_size=224, patch_size=16):
    """
    Adjusts ViViT position embeddings to fit VideoMAE model.

    Args:
        checkpoint (dict): The checkpoint dictionary containing ViViT weights.
        num_frames_vivit (int): Number of temporal slices ViViT was trained with (default 16).
        num_frames_vidmae (int): Number of temporal slices VideoMAE expects (default 8).
        img_size (int): Image size (default 224 for ViT-B/16).
        patch_size (int): Patch size (default 16 for ViT-B/16).

    Returns:
        Updated checkpoint with interpolated position embeddings.
    """
    pos_embed = checkpoint["encoder.pos_embed"]  # Shape: [1, 3137, 768] for ViViT-B (32x14x14 + 1 CLS token)

    # Remove CLS token
    cls_token, pos_embed = pos_embed[:, :1, :], pos_embed[:, 1:, :]
    # pos_embed.shape [1, 3136, 768] is now

    # Compute spatial token dimensions
    H = W = img_size // patch_size  # 224//16 = 14
    num_tokens_vivit = num_temp_vivit * H * W  # 16 * 14 * 14 = 3136
    num_tokens_vidmae = num_temp_vidmae * H * W  # 8 * 14 * 14 = 1568

    # Reshape position embeddings from [1, 3136, 768] → [1, 768, 16, 14, 14]
    pos_embed = pos_embed.reshape(1, num_temp_vivit, H, W, -1).permute(0, 4, 1, 2, 3)

    # Interpolate temporal dimension from 16 temporal slices to 8 temporal slices
    # [1, 768, 16, 14, 14] → [1, 768, 8, 14, 14]
    pos_embed = F.interpolate(pos_embed, size=(num_temp_vidmae, H, W), mode="trilinear", align_corners=False)

    # Flatten back to [1, 1568, 768]
    pos_embed = pos_embed.permute(0, 2, 3, 4, 1).reshape(1, num_tokens_vidmae, -1)

    # Assign back to checkpoint (VideoMAE doesn’t use CLS token)
    checkpoint["encoder.pos_embed"] = pos_embed

    return checkpoint


def _convert_vivit_to_videomae(vivit_ckpt_dict, vidmae_ckpt_dict=None):
    """
    Convert ViViT checkpoint keys to match VideoMAE keys.

    Args:
        vivit_ckpt_dict (dict): ViViT checkpoint dictionary.
        vidmae_ckpt_dict (dict): VideoMAE checkpoint dictionary.

    Returns:
        dict: Updated checkpoint dictionary with VideoMAE-compatible keys.
    """
    new_ckpt = {}

    # Mapping ViViT keys to VideoMAE keys
    key_map = {
        "embeddings.position_embeddings": "encoder.pos_embed",
        "embeddings.patch_embeddings.projection.weight": "encoder.patch_embed.proj.weight",
        "embeddings.patch_embeddings.projection.bias": "encoder.patch_embed.proj.bias",
        "layernorm.weight": "encoder.norm.weight",
        "layernorm.bias": "encoder.norm.bias",
    }

    # Convert encoder layers
    for i in range(12):  # Assuming 12 transformer blocks
        key_map.update({
            f"encoder.layer.{i}.layernorm_before.weight": f"encoder.blocks.{i}.norm1.weight",
            f"encoder.layer.{i}.layernorm_before.bias": f"encoder.blocks.{i}.norm1.bias",
            f"encoder.layer.{i}.attention.attention.query.weight": f"encoder.blocks.{i}.attn.qkv.weight",
            f"encoder.layer.{i}.attention.attention.query.bias": f"encoder.blocks.{i}.attn.q_bias",
            f"encoder.layer.{i}.attention.attention.key.weight": f"encoder.blocks.{i}.attn.qkv.weight",
            f"encoder.layer.{i}.attention.attention.key.bias": f"encoder.blocks.{i}.attn.v_bias",
            f"encoder.layer.{i}.attention.attention.value.weight": f"encoder.blocks.{i}.attn.qkv.weight",
            f"encoder.layer.{i}.attention.attention.value.bias": f"encoder.blocks.{i}.attn.v_bias",
            f"encoder.layer.{i}.attention.output.dense.weight": f"encoder.blocks.{i}.attn.proj.weight",
            f"encoder.layer.{i}.attention.output.dense.bias": f"encoder.blocks.{i}.attn.proj.bias",
            f"encoder.layer.{i}.layernorm_after.weight": f"encoder.blocks.{i}.norm2.weight",
            f"encoder.layer.{i}.layernorm_after.bias": f"encoder.blocks.{i}.norm2.bias",
            f"encoder.layer.{i}.intermediate.dense.weight": f"encoder.blocks.{i}.mlp.fc1.weight",
            f"encoder.layer.{i}.intermediate.dense.bias": f"encoder.blocks.{i}.mlp.fc1.bias",
            f"encoder.layer.{i}.output.dense.weight": f"encoder.blocks.{i}.mlp.fc2.weight",
            f"encoder.layer.{i}.output.dense.bias": f"encoder.blocks.{i}.mlp.fc2.bias",
        })

    for old_key, value in vivit_ckpt_dict.items():
        # Skip CLS token
        if "cls_token" in old_key:
            continue

        # Rename key if in mapping, otherwise keep original name
        new_key = key_map.get(old_key, old_key)

        new_ckpt[new_key] = value
        # Ensure the key exists in the target model
        if vidmae_ckpt_dict is not None and new_key not in vidmae_ckpt_dict:
            raise ValueError

    if "encoder.pos_embed" in new_ckpt:
        new_ckpt = adapt_vivit_pos_embed(new_ckpt)
    else:
        print("")

    return new_ckpt


def convert_vivit_to_videomae(vivit_ckpt_dict, vidmae_ckpt_dict=None):
    """
    Convert a ViViT checkpoint (with separate Q/K/V) to a VideoMAE-style checkpoint
    (with a single qkv layer, plus q_bias and v_bias).

    Args:
        vivit_ckpt_dict (dict): The raw ViViT checkpoint state dict.
        vidmae_ckpt_dict (dict): Optional reference dict for verifying that keys exist.

    Returns:
        dict: A new checkpoint dict compatible with VideoMAE's naming and shapes.
    """
    new_ckpt = {}

    # 1. Basic key map: positions, patch projection, final norms, MLP layers
    key_map = {
        "embeddings.position_embeddings": "encoder.pos_embed",
        "embeddings.patch_embeddings.projection.weight": "encoder.patch_embed.proj.weight",
        "embeddings.patch_embeddings.projection.bias":   "encoder.patch_embed.proj.bias",
        "layernorm.weight": "encoder.norm.weight",
        "layernorm.bias":   "encoder.norm.bias",
    }

    # 2. Map everything except Q/K/V from each block
    for i in range(12):  # Assuming a 12-block ViT-B
        # Layer norms, MLP, attn output
        key_map.update({
            f"encoder.layer.{i}.layernorm_before.weight": f"encoder.blocks.{i}.norm1.weight",
            f"encoder.layer.{i}.layernorm_before.bias":   f"encoder.blocks.{i}.norm1.bias",
            f"encoder.layer.{i}.attention.output.dense.weight": f"encoder.blocks.{i}.attn.proj.weight",
            f"encoder.layer.{i}.attention.output.dense.bias":   f"encoder.blocks.{i}.attn.proj.bias",
            f"encoder.layer.{i}.layernorm_after.weight":  f"encoder.blocks.{i}.norm2.weight",
            f"encoder.layer.{i}.layernorm_after.bias":    f"encoder.blocks.{i}.norm2.bias",
            f"encoder.layer.{i}.intermediate.dense.weight": f"encoder.blocks.{i}.mlp.fc1.weight",
            f"encoder.layer.{i}.intermediate.dense.bias":   f"encoder.blocks.{i}.mlp.fc1.bias",
            f"encoder.layer.{i}.output.dense.weight": f"encoder.blocks.{i}.mlp.fc2.weight",
            f"encoder.layer.{i}.output.dense.bias":   f"encoder.blocks.{i}.mlp.fc2.bias",
        })

    # ========== First pass: copy everything except {query,key,value} to new_ckpt ==========

    skip_keys = set()  # We'll gather Q/K/V in a second step
    for i in range(12):
        skip_keys.update({
            f"encoder.layer.{i}.attention.attention.query.weight",
            f"encoder.layer.{i}.attention.attention.query.bias",
            f"encoder.layer.{i}.attention.attention.key.weight",
            f"encoder.layer.{i}.attention.attention.key.bias",
            f"encoder.layer.{i}.attention.attention.value.weight",
            f"encoder.layer.{i}.attention.attention.value.bias",
        })

    for old_key, value in vivit_ckpt_dict.items():
        # Skip CLS token if present
        if "cls_token" in old_key:
            continue
        if old_key in skip_keys:
            continue

        new_key = key_map.get(old_key, old_key)
        new_ckpt[new_key] = value

        # (Optional) check if new_key is in reference
        if vidmae_ckpt_dict is not None and new_key not in vidmae_ckpt_dict:
            raise ValueError(f"Key {new_key} not found in target VideoMAE dict.")

    # ========== Second pass: gather Q/K/V and concatenate ==========

    for i in range(12):
        # 2A. Gather q/k/v weights
        q_w = vivit_ckpt_dict[f"encoder.layer.{i}.attention.attention.query.weight"]  # [768,768]
        k_w = vivit_ckpt_dict[f"encoder.layer.{i}.attention.attention.key.weight"]    # [768,768]
        v_w = vivit_ckpt_dict[f"encoder.layer.{i}.attention.attention.value.weight"]  # [768,768]
        qkv_w = torch.cat([q_w, k_w, v_w], dim=0)  # [3*768, 768] = [2304, 768]
        new_ckpt[f"encoder.blocks.{i}.attn.qkv.weight"] = qkv_w

        # 2B. Gather q/k/v biases
        q_b = vivit_ckpt_dict[f"encoder.layer.{i}.attention.attention.query.bias"]  # [768]
        k_b = vivit_ckpt_dict[f"encoder.layer.{i}.attention.attention.key.bias"]    # [768]
        v_b = vivit_ckpt_dict[f"encoder.layer.{i}.attention.attention.value.bias"]  # [768]

        # VideoMAE typically uses a bias-less qkv linear plus separate q_bias and v_bias,
        # or a similar pattern. Commonly, 'k' bias is unused. So we do:
        new_ckpt[f"encoder.blocks.{i}.attn.q_bias"] = q_b  # matches "q" portion
        new_ckpt[f"encoder.blocks.{i}.attn.v_bias"] = v_b  # matches "v" portion
        # k_b often remains unused or zeroed out, depending on the code.

    # ========== Lastly, adapt position embeddings if present ==========

    if "encoder.pos_embed" in new_ckpt:
        new_ckpt = adapt_vivit_pos_embed(new_ckpt)
    else:
        print("Warning: No position embedding found; skipping adaptation.")

    return new_ckpt



checkpoint_path = "../../logs/pretrained/vivit/vivit-b-16x2-kinetics400.pth"
model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", attn_implementation="sdpa", torch_dtype=torch.float16)

ckpt_dict = model.state_dict()
# print(ckpt_dict.keys())
# exit(0)

ckpt_dict_keys = list(ckpt_dict.keys())

with open(os.path.splitext(checkpoint_path)[0] + ".json", "w") as f:
    json.dump(ckpt_dict_keys, f, indent=2)

new_dict = convert_vivit_to_videomae(vivit_ckpt_dict=ckpt_dict)

ckpt_dict_keys = list(new_dict.keys())
with open(os.path.splitext(checkpoint_path)[0] + "_vidmae.json", "w") as f:
    json.dump(ckpt_dict_keys, f, indent=2)

ckpt_vidmae_path = "../../logs/pretrained/k400_vitb/checkpoint.pth"
ckpt_vidmae = torch.load(ckpt_vidmae_path, map_location='cpu')
ckpt_vidmae_keys = list(ckpt_vidmae["model"].keys())


pos_tokens_vivit = new_dict["encoder.pos_embed"]

diff = set(ckpt_dict_keys).symmetric_difference(ckpt_vidmae_keys)

torch.save({"model": new_dict}, os.path.splitext(checkpoint_path)[0] + "_vidmae.pth")

print("")
