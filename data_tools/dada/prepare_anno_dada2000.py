import pandas as pd
from natsort import natsorted

dada2k_anno = "/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA2000/DADA2K_my_split/orig_training.txt"
out_anno_train = "/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA2000/DADA2K_my_split/training.txt"
out_anno_val = "/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA2000/DADA2K_my_split/validation.txt"
full_anno = "/mnt/experiments/sorlova/datasets/LOTVS/DADA/DADA2000/annotation/full_anno.csv"

orig_all_clips = []
train_clips = []
val_clips = []

anno = pd.read_csv(full_anno)
for i, row in anno.iterrows():
    subdir = row["video"]
    catdir = row["type"]
    orig_all_clips.append(f"{str(int(catdir))}/{str(int(subdir)).zfill(3)}")
orig_all_clips = natsorted(orig_all_clips)

# Read the text file and parse each line
with open(dada2k_anno, "r") as file:
    for line in file:
        main_data, text = line.strip().split(",", 1)
        directory, label, start, end, toa = main_data.split()
        train_clips.append(directory)

train_clips = set(train_clips)
assert len(train_clips) == len(train_clips.intersection(orig_all_clips))
train_clips = natsorted(list(train_clips))

for clip in orig_all_clips:
    if clip in train_clips:
        continue
    else:
        val_clips.append(clip)

assert len(train_clips) + len(val_clips) == len(orig_all_clips)

with open(out_anno_train, "w") as f:
    f.write("\n".join(train_clips))
with open(out_anno_val, "w") as f:
    f.write("\n".join(val_clips))

print("Done!")

