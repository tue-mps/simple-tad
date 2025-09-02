import os
import pandas as pd
from natsort import natsorted

dada2k_anno = "/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/annotation/orig_full_testing-text.txt"
out_anno_train = "/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/CAPDATA_my_split/training.txt"
out_anno_val = "/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/CAPDATA_my_split/validation.txt"
full_anno = "/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/annotation/cap_annotation_file.csv"

orig_all_clips = []
train_clips = []
val_clips = []

os.makedirs("/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/CAPDATA_my_split", exist_ok=True)

anno = pd.read_csv(full_anno)
for i, row in anno.iterrows():
    subdir = str(row["video"]).strip()
    catdir = str(row["type"]).strip()
    orig_all_clips.append(f"{str(int(catdir))}/{str(int(subdir)).zfill(6)}")
orig_all_clips = natsorted(orig_all_clips)

# Read the text file and parse each line
with open(dada2k_anno, "r") as file:
    for line in file:
        main_data, text = line.strip().split(",", 1)
        directory, label, start, end, toa = main_data.split()
        val_clips.append(directory)

val_clips = set(val_clips)
print(f"all clips, first 5: {orig_all_clips[:5]}")
print(f"val clips, first 5: {list(val_clips)[:5]}")
print(f"val clips: {len(val_clips)}")
print(f"val clips in all anno: {len(val_clips.intersection(orig_all_clips))}")
assert len(val_clips) == len(val_clips.intersection(orig_all_clips))
val_clips = natsorted(list(val_clips))

for clip in orig_all_clips:
    if clip in val_clips:
        continue
    else:
        train_clips.append(clip)

assert len(train_clips) + len(val_clips) == len(orig_all_clips)

print(f"train clips: {len(train_clips)}")

with open(out_anno_train, "w") as f:
    f.write("\n".join(train_clips))
with open(out_anno_val, "w") as f:
    f.write("\n".join(val_clips))

print("Done!")

