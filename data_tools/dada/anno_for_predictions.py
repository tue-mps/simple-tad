import os
import pandas as pd
from tqdm import tqdm

from dada import FrameClsDataset_DADA


class MockArgs:
    def __init__(self):
        self.input_size = 224  # Example input size
        self.mask_type = 'tube'  # Masking type, 'tube' in this case
        self.window_size = (8, 14, 14)  # Example window size for TubeMaskingGenerator
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


dataset = FrameClsDataset_DADA(
    anno_path='DADA2K_my_split/validation.txt',
    data_path="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/DADA2K",
    mode="test",
    view_len=16,
    view_step=1,
    orig_fps=30,
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
#all_bin_labels = []
all_ego_labels = []
all_night_labels = []

for i in tqdm(range(L)):
    clip_id, frame_seq = dataset.dataset_samples[i]
    clip_name = dataset.clip_names[clip_id]
    last_seq_id = frame_seq[-1]
    last_ts = dataset.clip_timesteps[clip_id][last_seq_id]
    filename = f"{str(last_ts).zfill(4)}.png"
    cat_labels = dataset.clip_cat_labels[clip_id][last_seq_id]
    bin_labels = dataset.clip_bin_labels[clip_id][last_seq_id]
    ego_labels = dataset.clip_bin_labels[clip_id][last_seq_id]*dataset.clip_ego[clip_id]
    night_labels = dataset.clip_night[clip_id]
    #
    all_clip_names.append(clip_name)
    all_frame_names.append(filename)
    all_cat_labels.append(cat_labels)
    #all_bin_labels.append(bin_labels)
    all_ego_labels.append(ego_labels)
    all_night_labels.append(night_labels)

df = pd.DataFrame({
    "clip": all_clip_names,
    "filename": all_frame_names,
    "cat": all_cat_labels,
    "ego": all_ego_labels,
    "night": all_night_labels
})


df.to_csv("/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/DADA2K/DADA2K_my_split/frame_level_anno_val.csv")

print("Done!")

