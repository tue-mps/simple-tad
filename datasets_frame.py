import os
from torchvision import transforms

from transforms import *
from masking_generator import TubeMaskingGenerator
from dota import FrameClsDataset_DoTA, VideoMAE_DoTA
from dada import VideoMAE_DADA2K_prepared, FrameClsDataset_DADA
from bdd100k import VideoMAE_BDD100K_prepared
from kinetics import VideoMAE_aligned
from datasets import build_pretraining_dataset as orig_build_pre


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr
    

class DataAugmentationForVideoMAE_LightCrop(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, 1, 0.975, 0.95, 0.9, .875, 0.85])
        self.transform = transforms.Compose([    
            self.train_augmentation,                        
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(is_train, args):
    _transform = DataAugmentationForVideoMAE(args)
    _transform_like_finetune = DataAugmentationForVideoMAE_LightCrop(args)
    transform = _transform_like_finetune if args.transforms_finetune_align else _transform

    if args.data_set == 'DoTA':
        anno_path = None
        orig_fps = 10
        if is_train is True:
            mode = 'train'
            anno_path = 'all_split.txt' # 'train_split.txt'
        else:
            mode = 'validation'
            anno_path = 'all_split.txt' # 'val_split.txt'
        dataset = VideoMAE_DoTA(
            anno_path=anno_path,
            data_path=args.data_path,
            video_ext='mp4',
            is_color=True,
            view_len=args.num_frames,
            view_step=args.sampling_rate,
            orig_fps=orig_fps,
            target_fps=args.view_fps,
            transform=transform,
            temporal_jitter=False,
            video_loader=True,
            use_decord=True,
            lazy_init=False,
            args=args)
    elif args.data_set == 'DADA2K':
        anno_path = "DADA2K_my_split/all.txt"
        orig_fps = 30
        dataset = VideoMAE_DADA2K_prepared(
            anno_path=anno_path,
            data_path=args.data_path,
            video_ext='.png',
            is_color=True,
            view_len=args.num_frames,
            view_step=args.sampling_rate,
            orig_fps=orig_fps,
            target_fps=args.view_fps,
            transform=transform,
            temporal_jitter=False,
            video_loader=True,
            use_decord=True,
            lazy_init=False,
            args=args
        )
    elif args.data_set == 'BDD100K':
        droot = "/datasets/bdd100k_splits"
        if not os.path.exists(droot):
            raise FileNotFoundError("Please, provide the path to the folder 'bdd100k_splits' that contains splits! datasets_frame.py, line 121")
        orig_fps = 30
        if args.num_frames == 8 and args.view_fps == 5:
            clips_txt = os.path.join(droot, "prepared_views_8frames_5fps/all_clips.txt")
            views_txt = os.path.join(droot, "prepared_views_8frames_5fps/all_dataset_samples.txt")
        else:
            clips_txt = os.path.join(droot, "prepared_views/all_clips.txt")
            views_txt = os.path.join(droot, "prepared_views/all_dataset_samples.txt")
        dataset = VideoMAE_BDD100K_prepared(
            clips_txt=clips_txt,
            views_txt=views_txt,
            fps=orig_fps,
            target_fps=args.view_fps,
            root=args.data_path,
            setting=os.path.join(droot, "all.txt"),
            video_ext='mov',
            is_color=True,
            modality='rgb',
            new_length=args.num_frames,
            new_step=args.sampling_rate,
            transform=transform,
            temporal_jitter=False,
            video_loader=True,
            use_decord=True,
            lazy_init=False,
            args=args)
    elif args.data_set == 'CAP-DATA':
        anno_path = "CAPDATA_my_split/training.txt"
        droot = "/projects/0/prjs1424/sveta/RiskNetData/LOTVS-DADA/CAP-DATA"
        orig_fps = 30
        if args.num_frames == 8 and args.view_fps == 5:
            clips_txt=os.path.join(droot, "prepared_splits_8frames_5fps/training_clips.txt")
            timesteps_pkl=os.path.join(droot, "prepared_splits_8frames_5fps/training_timesteps.pkl")
            views_pkl=os.path.join(droot, "prepared_splits_8frames_5fps/training_dataset_samples.pkl")
        else:
            clips_txt=os.path.join(droot, "prepared_splits/training_clips.txt")
            timesteps_pkl=os.path.join(droot, "prepared_splits/training_timesteps.pkl")
            views_pkl=os.path.join(droot, "prepared_splits/training_dataset_samples.pkl")
        dataset = VideoMAE_DADA2K_prepared(
            clips_txt=clips_txt,
            timesteps_pkl=timesteps_pkl,
            views_pkl=views_pkl,
            setting=anno_path,
            root=args.data_path,
            video_ext='.jpg',
            is_color=True,
            new_length=args.num_frames,
            new_step=args.sampling_rate,
            fps=orig_fps,
            target_fps=args.view_fps,
            transform=transform,
            temporal_jitter=False,
            video_loader=True,
            use_decord=True,
            lazy_init=False,
            args=args
        )
    # elif args.data_set == 'K700_aligned':  # with our augmentation
    #     droot = '/scratch-nvme/ml-datasets/kinetics/k700-2020'
    #     dataset = VideoMAE_aligned(
    #         target_fps=args.view_fps,
    #         root=args.data_path,
    #         setting="annotations/train.csv",
    #         video_ext='mp4',
    #         is_color=True,
    #         modality='rgb',
    #         new_length=args.num_frames,
    #         new_step=args.sampling_rate,
    #         transform=transform,
    #         temporal_jitter=False,
    #         video_loader=True,
    #         use_decord=True,
    #         lazy_init=False,
    #         args=args)
    else:
        dataset = orig_build_pre(args)
    print("Data Aug = %s" % str(transform))
    return dataset


def build_frame_dataset(is_train, test_mode, args):
    if args.data_set.startswith('DoTA'):
        mode = None
        anno_path = None
        orig_fps = 10
        if is_train is True:
            mode = 'train'
            if "_half" in args.data_set:
                anno_path = 'half_train_split.txt'
            elif "_amnet" in args.data_set:
                anno_path = 'amnet_train_split300.txt'
            else: 
                anno_path = 'train_split.txt'
            sampling_rate = args.sampling_rate
        elif test_mode is True:
            mode = 'test'
            anno_path = 'val_split.txt'
            sampling_rate = 1 # args.sampling_rate_val if args.sampling_rate_val > 0 else args.sampling_rate
        else:  
            mode = 'validation'
            anno_path = 'val_split.txt'
            sampling_rate = args.sampling_rate_val if args.sampling_rate_val > 0 else args.sampling_rate

        dataset = FrameClsDataset_DoTA(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            view_len=args.num_frames,
            view_step=sampling_rate,
            orig_fps=orig_fps,  # for DoTA
            target_fps=args.view_fps,  # 10
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=1,  # 1
            num_crop=1,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            args=args)
        nb_classes = 2

    elif args.data_set.startswith('DADA2K'):
        mode = None
        anno_path = None
        orig_fps = 30
        if is_train is True:
            mode = 'train'
            anno_path = 'DADA2K_my_split/half_training.txt' if "_half" in args.data_set else "DADA2K_my_split/training.txt"
            sampling_rate = args.sampling_rate
        elif test_mode is True:
            mode = 'test'
            anno_path = "DADA2K_my_split/validation.txt"
            sampling_rate = args.sampling_rate_val if args.sampling_rate_val > 0 else args.sampling_rate
        else:
            mode = 'validation'
            anno_path = "DADA2K_my_split/validation.txt"
            sampling_rate = args.sampling_rate_val if args.sampling_rate_val > 0 else args.sampling_rate

        dataset = FrameClsDataset_DADA(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            view_len=args.num_frames,
            view_step=sampling_rate,
            orig_fps=orig_fps,  # original FPS of the dataset
            target_fps=args.view_fps,  # 10
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=1,  # 1
            num_crop=1,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            args=args)
        nb_classes = 2

    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
