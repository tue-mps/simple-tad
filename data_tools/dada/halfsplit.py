import os
import json
import random
from tqdm import tqdm
from natsort import natsorted

anno_path='DADA2K_my_split/training.txt'
data_path = "/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/DADA2K"

with open(os.path.join(data_path, anno_path), 'r') as file:
    clip_names = [line.rstrip() for line in file]

clips_by_cat = {}    
for clip in tqdm(clip_names, desc="Reading annotations"):
    cat, vid = clip.split("/")
    if cat in clips_by_cat:
        clips_by_cat[cat].append(clip)
    else:
        clips_by_cat[cat] = [clip]

chosen_clips = []
for cat in tqdm(clips_by_cat, desc="Splitting..."):
    cat_clips = clips_by_cat[cat]
    if len(cat_clips) == 1:
        chosen_clips.extend(cat_clips)
    if len(cat_clips) > 1:
        random.shuffle(cat_clips)
        split_idx = int(len(cat_clips) * 0.5)
        chosen_clips.extend(cat_clips[:split_idx])
print(f"Initial clips len: {len(clip_names)}, Chosen clips len: {len(chosen_clips)}")

with open(os.path.join(data_path, "DADA2K_my_split/half_training.txt"), 'w') as file:
    for clip in chosen_clips:
        file.write(f"{clip}\n")

