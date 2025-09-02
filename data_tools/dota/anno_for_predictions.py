import os
import pandas as pd
from tqdm import tqdm

import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from dota import FrameClsDataset_DoTA


class MockArgs:
    def __init__(self):
        self.input_size = 224  # Example input size
        self.mask_type = 'tube'  # Masking type, 'tube' in this case
        self.window_size = (8, 14, 14)  # Example window size for TubeMaskingGenerator, IN TOKENS
        self.mask_ratio = 0.90  # Example mask ratio
        self.transforms_finetune_align = True
        self.aa = 'rand-m3-n3-mstd0.5-inc1' # "rand-m7-n4-mstd0.5-inc1"
        self.train_interpolation = "bicubic"
        self.ttc_TT = 10.
        self.ttc_TA = 10.
        self.ttc_arTT = 10.
        self.ttc_arTA = 10.
        self.loss = "crossentropy"


args = MockArgs()


dataset = FrameClsDataset_DoTA(
    anno_path='val_split.txt',
    data_path="/mnt/experiments/sorlova/datasets/DoTA", # "/gpfs/work3/0/tese0625/RiskNetData/DoTA_refined",
    mode="test",
    view_len=16,
    view_step=1,
    orig_fps=10,
    target_fps=10,
    num_segment=1,
    test_num_segment=1,
    test_num_crop=1,
    num_crop=1,
    keep_aspect_ratio=True,
    crop_size=224,
    short_side_size=224,
    args=args,
)


L = len(dataset)

all_clip_names = []
all_frame_names = []
all_cat_labels = []
all_ego_labels = []
all_night_labels = []
all_clip_lvl_cat = []
all_clip_lvl_ego = []

for i in tqdm(range(L)):
    clip_id, frame_seq = dataset.dataset_samples[i]
    clip_name = dataset.clip_names[clip_id]
    last_seq_id = frame_seq[-1]
    last_ts = dataset.clip_timesteps[clip_id][last_seq_id]
    filename = f"{str(last_ts).zfill(6)}.jpg"
    cat_labels = dataset.clip_cat_labels[clip_id][last_seq_id]
    ego_labels = dataset.clip_bin_labels[clip_id][last_seq_id]*dataset.clip_ego[clip_id]
    night_labels = dataset.clip_night[clip_id]
    #
    clip_lvl_cat = dataset.clip_level_cat_labels[clip_id]
    clip_lvl_ego = dataset.clip_level_ego[clip_id]
    #
    all_clip_names.append(clip_name)
    all_frame_names.append(filename)
    all_cat_labels.append(cat_labels)
    all_ego_labels.append(ego_labels)
    all_night_labels.append(night_labels)
    #
    all_clip_lvl_cat.append(clip_lvl_cat)
    all_clip_lvl_ego.append(clip_lvl_ego)


df = pd.DataFrame({
    "clip": all_clip_names,
    "filename": all_frame_names,
    "cat": all_cat_labels,
    "ego": all_ego_labels,
    "night": all_night_labels,
    "clip_lvl_cat": all_clip_lvl_cat,
    "clip_lvl_ego": all_clip_lvl_ego,
})


df.to_csv(os.path.join(dataset.data_path, "dataset", "frame_level_anno_val.csv"))

print("Done!")

