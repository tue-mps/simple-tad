import os
from bdd100k import VideoMAE_BDD100K

class MockArgs:
    def __init__(self):
        self.input_size = 224  # Example input size
        self.mask_type = 'tube'  # Masking type, 'tube' in this case
        self.window_size = (8, 14, 14)  # Example window size for TubeMaskingGenerator
        self.mask_ratio = 0.90  # Example mask ratio
        self.transforms_finetune_align = True

args = MockArgs()

dataset = VideoMAE_BDD100K(
    root="/scratch-nvme/ml-datasets/bdd100k/videos",
    setting="/gpfs/work3/0/tese0625/datasets/bdd100k_splits/all.txt",
    video_ext='mov',
    is_color=True,
    modality='rgb',
    new_length=8,
    target_fps=5,
    new_step=4,
    transform=None,
    temporal_jitter=False,
    video_loader=True,
    use_decord=True,
    lazy_init=False,
    manager=None,
    args=args)
L = len(dataset)
print(f"\nLength of the dataset: {L}")

clips = dataset.clips
samples = dataset.dataset_samples

os.makedirs("/gpfs/work3/0/tese0625/datasets/bdd100k_splits/prepared_views_8frames_5fps/", exist_ok=True)

print("Writing clips...")
with open("/gpfs/work3/0/tese0625/datasets/bdd100k_splits/prepared_views_8frames_5fps/all_clips.txt", "w") as file:
    for line in clips:
        file.write(line + "\n")
print("\tClips done!")

print("Writing samples...")
with open("/gpfs/work3/0/tese0625/datasets/bdd100k_splits/prepared_views_8frames_5fps/all_dataset_samples.txt", "w") as file:
    for s in samples:
        file.write(f"{s[0]},{s[1]}\n")
print("\tSamples done!")

print("Done!")