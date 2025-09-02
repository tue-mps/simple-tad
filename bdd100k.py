import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
#from numpy.lib.function_base import disp
import torch
import decord
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
import video_transforms as video_transforms 
import volume_transforms as volume_transforms

from kinetics import VideoMAE
from dataset.sequencing import RegularSequencer, RegularSequencerWithStart


video_ext = (".mov", ".mp4", ".avi", ".mkv")
ignore_videos = ("val/c4742900-81aa45ae.mov",)


class VideoMAE_BDD100K(VideoMAE):

    def __init__(self, fps=30, target_fps=10, **kwargs):
        self.fps = fps
        self.tfps = target_fps
        super().__init__(**kwargs)
        self.sequencer = RegularSequencerWithStart(seq_frequency=self.tfps, seq_length=self.new_length, step=self.new_step)
        self._prepare_views()
        if self.args.transforms_finetune_align:
            self._getitem = self._getitem_finetune_align
        else:
            self._getitem = self._getitem_orig

    def _prepare_views(self):
        dataset_sequences = []
        N = len(self.clips)
        for i in tqdm(range(N), desc=f"Preparing views of len {self.new_length} with FPS {self.tfps}"):
            decord_vr = decord.VideoReader(os.path.join(self.root, self.clips[i]), num_threads=1)
            duration = len(decord_vr)
            sequences = self.sequencer.get_sequences(timesteps_nb=duration, input_frequency=self.fps)
            if sequences is None:
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
        self.dataset_samples = dataset_sequences

    def _getitem_orig(self, index):
        sample = self.dataset_samples[index]
        clip_id, frame_seq = sample
        clip_name = self.clips[clip_id]
        video_name = os.path.join(self.root, clip_name)

        if self.video_loader:
            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
            images = self.decord_extract_frames(video_reader=decord_vr, frame_id_list=frame_seq, duration=duration, video_name=video_name)
            assert len(images) > 0

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0
                                                                                                    ,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data, mask)
    
    def _getitem_finetune_align(self, index):
        sample = self.dataset_samples[index]
        clip_id, frame_seq = sample
        clip_name = self.clips[clip_id]
        video_name = os.path.join(self.root, clip_name)

        if self.video_loader:
            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
            images = self.decord_extract_frames_cv2(video_reader=decord_vr, frame_id_list=frame_seq, duration=duration, video_name=video_name)
            assert len(images) > 0

        # augment
        images = self._aug_frame(images, self.args)

        process_data, mask = self.transform((images, None)) # T*C,H,W

        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0
                                                                                                    ,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data, mask)
    
    def _aug_frame(
        self,
        buffer,
        args,
    ):
        h, w, _ = buffer[0].shape
        # first, resize to a bit larger size (e.g. 320) instead of the target one (224)
        min_side = min(h, w, self.intermediate_size)
        do_pad = video_transforms.pad_wide_clips(h, w, min_side)
        buffer = [do_pad(img) for img in buffer]
        if torch.rand(1).item() > 0.3:
            aug_transform = video_transforms.create_random_augment(
                input_size=(args.input_size, args.input_size),
                auto_augment=args.aa,
                interpolation=args.train_interpolation,
                do_transforms=video_transforms.DRIVE_TRANSFORMS
            )
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
            buffer = aug_transform(buffer)
        else:
            buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        return buffer
    
    def __getitem__(self, index):
        return self._getitem(index)
    

    def __getitem__check_clips(self, index):
        clip_name = self.clips[index]
        video_name = os.path.join(self.root, clip_name)

        #if self.video_loader:
        try:
            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
            segment_indices, skip_offsets = self._sample_train_indices(duration)
            images = self._video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)
            assert len(images) > 0
            return 1
        except Exception as e:
            print(f"Decord failed for {video_name} with error: {e}.", end="")
            self.corrupt_clips_decord.append(clip_name)
            print(f" Falling back to OpenCV.")
            # Fall back to OpenCV
            cap = cv2.VideoCapture(video_name)
            if not cap.isOpened():
                #raise RuntimeError(f"Error: Unable to open video file {video_name}")
                self.corrupt_clips.append(clip_name)
                return 0
            duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if duration < self.new_length:
                self.corrupt_clips.append(clip_name)
            return 0

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0
                                                                                                    ,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data, mask)

    def __len__(self):
        return len(self.dataset_samples)

    def _make_dataset_snellius(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting, "r") as split_f:
            clips = [line.strip() for line in split_f]
        for iv in ignore_videos:
            assert iv in clips
            clips.remove(iv)
        assert len(clips) > 0, f"Cannot find any video clips for the given split: {setting}"
        return clips


class VideoMAE_BDD100K_prepared(VideoMAE_BDD100K):

    def __init__(self, clips_txt, views_txt, **kwargs):
        self.clips_txt = clips_txt
        self.views_txt = views_txt
        super().__init__(**kwargs)

    def _make_dataset_snellius(self, directory, setting):
        clips = []
        # read from the file
        with open(self.clips_txt, 'r') as file:
            clips = [line.rstrip() for line in file]
        assert len(clips) > 0, f"Cannot find any video clips for the given split: {setting}"
        return clips
    
    def _prepare_views(self):
        dataset_sequences = []
        # read from the file
        with open(self.views_txt, 'r') as file:
            for line in file:
                el1, el2 = line.strip().split(",", 1)
                el1 = int(el1.strip())
                el2 = [int(x.strip()) for x in el2.strip('[]').split(',')]
                dataset_sequences.append([el1, el2])
        self.dataset_samples = dataset_sequences


class MockArgs:
    def __init__(self):
        self.input_size = 224  # Example input size
        self.mask_type = 'tube'  # Masking type, 'tube' in this case
        self.window_size = (8, 14, 14)  # Example window size for TubeMaskingGenerator
        self.mask_ratio = 0.90  # Example mask ratio
        self.transforms_finetune_align = True


class CustomDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize attributes to store corrupt clips
        self.corrupt_clips = []
        self.corrupt_clips_decord = []


if __name__ == "__main__":
    from datasets import DataAugmentationForVideoMAE
    from multiprocessing import Manager
    manager = Manager()
    args = MockArgs()
    tf = DataAugmentationForVideoMAE(args)

    if False:
        bdd_path = "/scratch-nvme/ml-datasets/bdd100k/videos"
        subfolder = ("train", "val", "test")
        out_path = "/projects/0/prjs1424/sveta/datasets/bdd100k_splits/all.txt"
        filelist = []
        if True:
            for i, sf in enumerate(subfolder):
                print(f"{i+1}/{len(subfolder)} subset! {sf}")
                clips = os.listdir(os.path.join(bdd_path, sf))
                clips = [os.path.join(sf, c) for c in clips if os.path.splitext(c)[1] in video_ext]
                print(f"\t\tFound {len(clips)} videos!")
                filelist.extend(clips)
            print(f"Total clips found: {len(filelist)}")
            print(filelist[0])
            print(filelist[75000])
            print(filelist[85000])
            with open(out_path, "w") as f:
                f.write("\n".join(filelist))
            print(f"Saved the list as {out_path}")
        # 
        print(f"\n ... Read and check")
        with open(out_path, "r") as f:
            filelist2 = [line.strip() for line in f]
        print(f"Same lists? {filelist == filelist2}")
        exit(0)
        #

    if False:
        dataset = VideoMAE_BDD100K(
        root="/scratch-nvme/ml-datasets/bdd100k/videos",
        setting="/gpfs/work3/0/tese0625/datasets/bdd100k_splits/all.txt",
        video_ext='mov',
        is_color=True,
        modality='rgb',
        new_length=16,
        new_step=130,
        fps=30, 
        target_fps=10,
        transform=tf,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
        L = len(dataset)
        print(f"Dataset length: {L} for new_step {dataset.new_step}")
        item = dataset[0]
        exit(0)
    
    print("Start checking the video clips!")
    droot = "/projects/0/prjs1424/sveta/datasets"
    
    dataset = VideoMAE_BDD100K(
        root="/scratch-nvme/ml-datasets/bdd100k/videos",
        setting=os.path.join(droot, "bdd100k_splits/all.txt"),
        video_ext='mov',
        is_color=True,
        modality='rgb',
        new_length=16,
        new_step=4,
        transform=tf,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        manager=manager)
    L = len(dataset)
    
    dataloader = CustomDataLoader(dataset, batch_size=200, shuffle=False, num_workers=15, pin_memory=False, drop_last=False, persistent_workers=False)
    L2 = len(dataloader)
    print_break = L2 // 2
    print(f"\nDataset length: {L}, Batch size: 200, Batch numbers: {L2}, print break every {print_break} batches\n")

    for idx, batch in tqdm(enumerate(dataloader), total=L2, desc="Validating dataset"):
        _ = batch

        if idx % print_break == 0:
            problems = dataloader.dataset.corrupt_clips_decord
            problems_ = "\n".join(problems)
            if len(problems) > 0:
                with open(os.path.join(droot, f"bdd100k_splits/decord_err_train_b200_{idx}.txt"), mode="w") as f:
                    f.write(problems_)
            print(f"NB of corrupted videos found so far (decord): {len(problems)}")
            problems = dataloader.dataset.corrupt_clips
            problems_ = "\n".join(problems)
            if len(problems) > 0:
                with open(os.path.join(droot, f"bdd100k_splits/opencv_err_train_b200_{idx}.txt"), mode="w") as f:
                    f.write(problems_)
            print(f"NB of corrupted videos found so far (decord AND opencv): {len(problems)}")
        
    problems = dataloader.dataset.corrupt_clips_decord
    problems_ = "\n".join(problems)
    if len(problems) > 0:
        with open(os.path.join(droot, f"bdd100k_splits/decord_err_train_b200.txt"), mode="w") as f:
            f.write(problems_)
    problems = dataloader.dataset.corrupt_clips
    problems_ = "\n".join(problems)
    if len(problems) > 0:
        with open(os.path.join(droot, f"/gpfs/work3/0/tese0625/datasets/bdd100k_splits/opencv_err_train_b200.txt"), mode="w") as f:
            f.write(problems_)

    print("Done!")
    exit(0)


