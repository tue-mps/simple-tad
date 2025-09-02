import os
import cv2
import numpy as np
import torch
from natsort import natsorted
from timm.models import create_model

from flash_attention_class import FlashAttention  # You can comment this out
import modeling_finetune


IMG_EXT = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG")


def prepare_image(img, mean, std, inplace=True):
    if not (len(img.shape) == 3):
        raise TypeError(f'Input must be a 3D image tensor (C, H, W), but got shape: {img.shape}')

    if not inplace:
        img = np.copy(img)

    # We need RGB, (C, H, W), value range [0, 1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    img = img.float().div_(255.0)

    # Normalize with ImageNet mean and std
    dtype = img.dtype
    device = img.device
    mean = torch.as_tensor(mean, dtype=dtype, device=device).view(-1, 1, 1)
    std  = torch.as_tensor(std,  dtype=dtype, device=device).view(-1, 1, 1)
    img.sub_(mean).div_(std)

    return img


def main(ckpt_file, frames_folder, model_name='vit_small_patch16_224', with_flash_attn=False):
    device = torch.device("cuda")

    model = create_model(
        model_name=model_name,
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
        use_flash_attn=with_flash_attn
    )
    model.default_cfg = {
        'url': "",
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406),  
        'std': (0.229, 0.224, 0.225),
    }

    checkpoint = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Prepare the sliding window and read the frames
    sliding_window_frames = torch.zeros((1, 3, 16, 224, 224), dtype=torch.float32)
    # fill the first sliding window
    imglist = natsorted([item for item in os.listdir(frames_folder) if os.path.splitext(item)[1] in IMG_EXT])
    assert len(imglist) > 15, "We need at least 16 frames!"
    for i, img_file in enumerate(imglist[:16]):
        img = cv2.imread(os.path.join(frames_folder, img_file))
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        img = prepare_image(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        sliding_window_frames[0, :, i, :, :] = img
    sliding_window_frames = sliding_window_frames.to(device, non_blocking=True)
    # make first prediction
    out = model(sliding_window_frames)
    print(f"First prediction: {out}")

    # now can predict frame by frame
    for i, img_file in enumerate(imglist[16:]):
        if i < 16:
            continue

        # Prepare new frame
        img = cv2.imread(os.path.join(frames_folder, img_file))
        img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        img = prepare_image(img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        img = img.to(device, non_blocking=True)

        # Shift sliding window
        #   # drop the oldest frame and shift left
        sliding_window_frames = sliding_window_frames[:, :, 1:, :, :]  # shape: (1, 3, T-1, H, W)
        #   # add new frame to the end
        img = img.unsqueeze(0).unsqueeze(2)  # shape: (1, 3, 1, H, W)
        #   # concatenate along time dimension
        sliding_window_frames = torch.cat([sliding_window_frames, img], dim=2)  # (1, 3, T, H, W)

        # Make new prediction. The model does not apply softmax.
        raw_out = model(sliding_window_frames)
        risk_logit = raw_out[0][1].item()
        # We apply it manually to get probabilities:
        softmax_out = torch.nn.functional.softmax(raw_out)
        risk_prob = softmax_out[0][1].item()
        print(f"Frame {i}, risk logit: {risk_logit:.2f}, risk prob: {risk_prob:.2f}")

    print("Done!")


if __name__ == "__main__":
    main(
        ckpt_file="/mnt/adas7tb/sorlova/VideoMAE_logs/train_logs/finetune/DoTA_orig/vm1-small-dapt_dota_lr1e3_b56x1_dsampl1val1_ld06_aam6n3/checkpoint-bestmccauc.pth", #"your_checkpoint_file.pth",
        frames_folder="/mnt/adas7tb/sorlova/datasets/TSTTC/test/2023-01-03-09-19-35-051234/cam1", #"your_frames_folder",
        model_name="vit_small_patch16_224"
        )

