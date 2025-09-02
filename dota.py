import os
import zipfile
import cv2
import numpy as np
import torch
import pandas as pd
import json
from natsort import natsorted
from PIL import Image
from torchvision import transforms
import warnings
from torch.utils.data import Dataset

from functional import crop_clip
from random_erasing import RandomErasing
import video_transforms as video_transforms 
import volume_transforms as volume_transforms

from dataset.sequencing import RegularSequencer, RegularSequencerWithStart
from dataset.data_utils import smooth_labels, compute_time_vector


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CENTER_MEAN = [0.5, 0.5, 0.5]
CENTER_STD = [0.5, 0.5, 0.5]

# CENTER for PE, IMAGENET otherwise
MEAN = IMAGENET_MEAN
STD = IMAGENET_STD

meta_cat2code_ = {
    "start_stop_or_stationary":  "ST",
    "moving_ahead_or_waiting":   "AH",
    "lateral":                   "LA",
    "oncoming":                  "OC",
    "turning":                   "TC",
    "pedestrian":                "VP",
    "obstacle":                  "VO",
    "leave_to_left":             "OO_l",  # unique code
    "leave_to_right":            "OO_r",  # unique code
    "unknown":                   "UK"
}

meta_cat2code = {
    "start_stop_or_stationary":  "ST",
    "moving_ahead_or_waiting":   "AH",
    "lateral":                   "LA",
    "oncoming":                  "OC",
    "turning":                   "TC",
    "pedestrian":                "VP",
    "obstacle":                  "VO",
    "leave_to_left":             "OO",  # the same code
    "leave_to_right":            "OO",  # the same code
    "unknown":                   "UK"
}

meta_cat2id = {
    "start_stop_or_stationary":  1,
    "moving_ahead_or_waiting":   2,
    "lateral":                   3,
    "oncoming":                  4,
    "turning":                   5,
    "pedestrian":                6,
    "obstacle":                  7,
    "leave_to_right":            8,
    "leave_to_left":             9,
    "unknown":                   10
}

class FrameClsDataset_DoTA(Dataset):
    meta_path = {"val_split.txt": "metadata_val.json", "train_split.txt": "metadata_train.json"}
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train',
                 view_len=8, target_fps=10, orig_fps=10, view_step=10,
                 crop_size=224, short_side_size=320,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=1, test_num_crop=1, args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.view_len = view_len
        self.target_fps = target_fps
        self.orig_fps = orig_fps
        self.view_step = view_step
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        #self.new_height = new_height
        #self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.ttc_TT = args.ttc_TT if hasattr(args, "ttc_TT") else 2.
        self.ttc_TA = args.ttc_TA if hasattr(args, "ttc_TA") else 1.
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True

        self._read_anno()
        self._prepare_views()
        assert len(self.dataset_samples) > 0
        assert len(self._label_array) > 0

        if self.args.loss in ("2bce",):
            self.label_array = self._smoothed_label_array
        else:
            self.label_array = self._label_array

        count_safe = self._label_array.count(0)
        count_risk = self._label_array.count(1)
        print(f"\n\n===\n[{mode}] | COUNT safe: {count_safe}\nCOUNT risk: {count_risk}\n===")

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=MEAN,
                                           std=STD)
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(self.crop_size, self.crop_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=MEAN,
                                           std=STD)
            ])
            self.test_seg = [(0, 0)]
            self.test_dataset = self.dataset_samples
            self.test_label_array = self.label_array

    def _read_anno(self):
        clip_names = None
        clip_timesteps = []
        clip_binary_labels = []
        clip_cat_labels = []
        clip_ego = []
        clip_night = []
        clip_ttc = []
        clip_smoothed_labels = []

        with open(os.path.join(self.data_path, "dataset", self.anno_path), 'r') as file:
            clip_names = [line.rstrip() for line in file]
        for clip in clip_names:
            clip_anno_path = os.path.join(self.data_path, "dataset", "annotations", f"{clip}.json")
            with open(clip_anno_path) as f:
                anno = json.load(f)
                # sort is not required since we read already sorted timesteps from annotations
                timesteps = natsorted([int(os.path.splitext(os.path.basename(frame_label["image_path"]))[0]) for frame_label
                                  in anno["labels"]])
                cat_labels = [int(frame_label["accident_id"]) for frame_label in anno["labels"]]
                if_ego = anno["ego_involve"]
                if_night = anno["night"]
            binary_labels = [1 if l > 0 else 0 for l in cat_labels]
            ttc = compute_time_vector(binary_labels, fps=self.orig_fps, TT=self.ttc_TT, TA=self.ttc_TA)
            smoothed_labels = smooth_labels(labels=torch.Tensor(binary_labels), time_vector=ttc, before_limit=self.ttc_TT, after_limit=self.ttc_TA)

            clip_timesteps.append(timesteps)
            clip_binary_labels.append(binary_labels)
            clip_cat_labels.append(cat_labels)
            clip_ego.append(if_ego)
            clip_night.append(if_night)
            clip_ttc.append(ttc)
            clip_smoothed_labels.append(smoothed_labels)

        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels) == len(clip_cat_labels)
        self.clip_names = clip_names
        self.clip_timesteps = clip_timesteps
        self.clip_bin_labels = clip_binary_labels
        self.clip_cat_labels = clip_cat_labels
        self.clip_ego = clip_ego
        self.clip_night = clip_night
        self.clip_ttc = clip_ttc
        self.clip_smoothed_labels = clip_smoothed_labels

        # try to get clip-level category
        if self.anno_path in self.meta_path:
            with open(os.path.join(self.data_path, "dataset", self.meta_path[self.anno_path]), 'r') as file:
                meta_data = json.load(file)
            clip_level_cat_labels = []
            clip_level_ego = []
            for clip_name in clip_names:
                anomaly_class_str = meta_data[clip_name]["anomaly_class"]
                group, category = anomaly_class_str.split(": ")
                clip_level_ego.append(group == "ego")
                clip_level_cat_labels.append(meta_cat2code[category])
            self.clip_level_cat_labels = clip_level_cat_labels
            self.clip_level_ego = clip_level_ego
        else:
            self.clip_level_cat_labels = None
            self.clip_level_ego = None

    def _prepare_views(self):
        dataset_sequences = []
        label_array = []
        ttc = []
        smoothed_label_array = []
        sequencer = RegularSequencer(seq_frequency=self.target_fps, seq_length=self.view_len, step=self.view_step)
        N = len(self.clip_names)
        for i in range(N):
            timesteps = self.clip_timesteps[i]
            sequences = sequencer.get_sequences(timesteps_nb=len(timesteps), input_frequency=self.orig_fps)
            if sequences is None:
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
            label_array.extend([self.clip_bin_labels[i][seq[-1]] for seq in sequences])
            smoothed_label_array.extend([self.clip_smoothed_labels[i][seq[-1]] for seq in sequences])
            ttc.extend([self.clip_ttc[i][seq[-1]] for seq in sequences])
        self.dataset_samples = dataset_sequences
        self._label_array = label_array
        self.ttc = ttc
        self._smoothed_label_array = smoothed_label_array

    def __getitem__(self, index):

        if self.mode == 'train':
            args = self.args
            sample = self.dataset_samples[index]
            buffer, _, __ = self.load_images(sample, final_resize=False, resize_scale=1.)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, final_resize=False, resize_scale=1.)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                smoothed_label_list = []
                index_list = []
                ttc_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self._label_array[index]
                    smoothed_label = self._smoothed_label_array[index]
                    ttc = self.ttc[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    smoothed_label_list.append(smoothed_label)
                    index_list.append(index)
                    ttc_list.append(ttc)
                extra_info = [{"ttc": ttc_item, "smoothed_labels": slab_item} for ttc_item, slab_item in zip(ttc_list, smoothed_label_list)]
                return frame_list, label_list, index_list, extra_info
            else:
                buffer = self._aug_frame(buffer, args)
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index]}
            return buffer, self._label_array[index], index, extra_info

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer, _, __ = self.load_images(sample, final_resize=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index]}
            return buffer,self._label_array[index], index, extra_info

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            buffer, clip_name, frame_name = self.load_images(sample, final_resize=True)
            while len(buffer) == 0:
                warnings.warn("video {} not found during testing".format(str(self.test_dataset[index])))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                buffer, clip_name, frame_name = self.load_images(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            extra_info = {"ttc": self.ttc[index], "clip": clip_name, "frame": frame_name, "smoothed_labels": self._smoothed_label_array[index]}
            return buffer, self._label_array[index], index, extra_info
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):
        h, w, _ = buffer[0].shape
        # Perform data augmentation - vertical padding and horizontal flip
        # add padding
        do_pad = video_transforms.pad_wide_clips(h, w, self.crop_size)
        buffer = [do_pad(img) for img in buffer]

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            do_transforms=video_transforms.DRIVE_TRANSFORMS
        )

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)
        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer) # T C H W
        buffer = buffer.permute(0, 2, 3, 1) # T H W C
        # T H W C 
        buffer = tensor_normalize(
            buffer, MEAN, STD
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                max_area=0.1,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_images(self, dataset_sample, final_resize=False, resize_scale=None):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(6)}.jpg" for ts in timesteps]
        view = []
        with zipfile.ZipFile(os.path.join(self.data_path, "frames", clip_name, "images.zip"), 'r') as zipf:
            for fname in filenames:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                if final_resize:
                    img = cv2.resize(img, dsize=(self.crop_size, self.crop_size), interpolation=cv2.INTER_CUBIC)
                elif resize_scale is not None:
                    short_side = min(min(img.shape[:2]), self.short_side_size)
                    target_side = self.crop_size * resize_scale
                    k = target_side / short_side
                    img = cv2.resize(img, dsize=(0,0), fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
                else:
                    raise ValueError
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

                view.append(img)
        #view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=1,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


class VideoMAE_DoTA(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    Parameters
    ----------
    root : str, required.
        Path to the root folder storing the dataset.
    setting : str, required.
        A text file describing the dataset, each line per video sample.
        There are three items in each line: (1) video path; (2) video length and (3) video label.
    train : bool, default True.
        Whether to load the training or validation set.
    test_mode : bool, default False.
        Whether to perform evaluation on the test set.
        Usually there is three-crop or ten-crop evaluation strategy involved.
    name_pattern : str, default None.
        The naming pattern of the decoded video frames.
        For example, img_00012.jpg.
    video_ext : str, default 'mp4'.
        If video_loader is set to True, please specify the video format accordinly.
    is_color : bool, default True.
        Whether the loaded image is color or grayscale.
    modality : str, default 'rgb'.
        Input modalities, we support only rgb video frames for now.
        Will add support for rgb difference image and optical flow image later.
    num_segments : int, default 1.
        Number of segments to evenly divide the video into clips.
        A useful technique to obtain global video-level information.
        Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016.
    num_crop : int, default 1.
        Number of crops for each image. default is 1.
        Common choices are three crops and ten crops during evaluation.
    new_length : int, default 1.
        The length of input video clip. Default is a single image, but it can be multiple video frames.
        For example, new_length=16 means we will extract a video clip of consecutive 16 frames.
    new_step : int, default 1.
        Temporal sampling rate. For example, new_step=1 means we will extract a video clip of consecutive frames.
        new_step=2 means we will extract a video clip of every other frame.
    temporal_jitter : bool, default False.
        Whether to temporally jitter if new_step > 1.
    video_loader : bool, default False.
        Whether to use video loader to load data.
    use_decord : bool, default True.
        Whether to use Decord video loader to load data. Otherwise use mmcv video loader.
    transform : function, default None.
        A function that takes data and label and transforms them.
    data_aug : str, default 'v1'.
        Different types of data augmentation auto. Supports v1, v2, v3 and v4.
    lazy_init : bool, default False.
        If set to True, build a dataset instance without loading any dataset.
    """

    def __init__(self,
                 anno_path,
                 data_path,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 view_len=1,
                 view_step=1,
                 orig_fps=10,
                 target_fps=10,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False,
                 short_size=320,
                 args=None
                 ):

        super(VideoMAE_DoTA, self).__init__()
        self.anno_path = anno_path
        self.data_path = data_path
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.view_len = view_len
        self.view_step = view_step
        self.ofps = orig_fps
        self.tfps = target_fps
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init
        self.short_size = short_size
        self.ttc_TT = args.ttc_TT if hasattr(args, "ttc_TT") else 2.
        self.ttc_TA = args.ttc_TA if hasattr(args, "ttc_TA") else 1.
        self.sequencer = RegularSequencerWithStart(seq_frequency=self.tfps, seq_length=self.view_len, step=self.view_step)

        if not self.lazy_init:
            self._read_anno()
            self._prepare_views()
            if len(self.dataset_samples) == 0:
                raise RuntimeError("Found 0 video clips in subfolders of: " + data_path)
            
        if args.transforms_finetune_align:
            self._getitem = self._getitem_finetune_align
        else:
            self._getitem = self._getitem_orig

    def _read_anno(self):
        clip_names = None
        clip_timesteps = []
        clip_binary_labels = []
        clip_cat_labels = []
        clip_ego = []
        clip_night = []
        clip_ttc_out = []
        clip_ttc_in = []
        clip_smoothed_labels = []

        with open(os.path.join(self.data_path, "dataset", self.anno_path), 'r') as file:
            clip_names = [line.rstrip() for line in file]
        for clip in clip_names:
            clip_anno_path = os.path.join(self.data_path, "dataset", "annotations", f"{clip}.json")
            with open(clip_anno_path) as f:
                anno = json.load(f)
                # sort is not required since we read already sorted timesteps from annotations
                timesteps = natsorted([int(os.path.splitext(os.path.basename(frame_label["image_path"]))[0]) for frame_label
                                  in anno["labels"]])
                cat_labels = [int(frame_label["accident_id"]) for frame_label in anno["labels"]]
                if_ego = anno["ego_involve"]
                if_night = anno["night"]
            binary_labels = [1 if l > 0 else 0 for l in cat_labels]
            ttc_out = compute_time_vector(binary_labels, fps=self.ofps, TT=self.ttc_TT, TA=self.ttc_TA)
            smoothed_labels = smooth_labels(labels=torch.Tensor(binary_labels), time_vector=ttc_out, before_limit=self.ttc_TT, after_limit=self.ttc_TA)

            clip_timesteps.append(timesteps)
            clip_binary_labels.append(binary_labels)
            clip_cat_labels.append(cat_labels)
            clip_ego.append(if_ego)
            clip_night.append(if_night)
            clip_ttc_out.append(ttc_out)
            clip_smoothed_labels.append(smoothed_labels)

        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels) == len(clip_cat_labels)
        self.clip_names = clip_names
        self.clip_timesteps = clip_timesteps
        self.clip_bin_labels = clip_binary_labels
        self.clip_cat_labels = clip_cat_labels
        self.clip_ego = clip_ego
        self.clip_night = clip_night
        self.clip_ttc_out = clip_ttc_out
        self.clip_ttc_in = None
        self.clip_smoothed_labels = clip_smoothed_labels

    def _prepare_views(self):
        dataset_sequences = []
        label_array = []
        ttc = []
        smoothed_label_array = []
        sequencer = RegularSequencer(seq_frequency=self.tfps, seq_length=self.view_len, step=self.view_step)
        N = len(self.clip_names)
        for i in range(N):
            timesteps = self.clip_timesteps[i]
            sequences = sequencer.get_sequences(timesteps_nb=len(timesteps), input_frequency=self.ofps)
            if sequences is None:
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
            label_array.extend([self.clip_bin_labels[i][seq[-1]] for seq in sequences])
            smoothed_label_array.extend([self.clip_smoothed_labels[i][seq[-1]] for seq in sequences])
            ttc.extend([self.clip_ttc[i][seq[-1]] for seq in sequences])
        self.dataset_samples = dataset_sequences
        self._label_array = label_array
        self.ttc = ttc
        self._smoothed_label_array = smoothed_label_array

    def load_images(self, dataset_sample, short_size=320):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(6)}.jpg" for ts in timesteps]
        view = []
        with zipfile.ZipFile(os.path.join(self.data_path, "frames", clip_name, "images.zip"), 'r') as zipf:
            for fname in filenames:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                # resze
                if short_size is not None:
                    h, w, _ = img.shape
                    if h < w:
                        scale = 320 / h
                        new_h, new_w = 320, int(w * scale)
                    else:
                        scale = 320 / w
                        new_h, new_w = int(h * scale), 320
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                view.append(img)
        #view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]
    
    def load_images_cv2(self, dataset_sample, short_size=320):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(6)}.jpg" for ts in timesteps]
        view = []
        with zipfile.ZipFile(os.path.join(self.data_path, "frames", clip_name, "images.zip"), 'r') as zipf:
            for fname in filenames:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                img = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
                # resze
                if short_size is not None:
                    h, w, _ = img.shape
                    if h < w:
                        scale = 320 / h
                        new_h, new_w = 320, int(w * scale)
                    else:
                        scale = 320 / w
                        new_h, new_w = int(h * scale), 320
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                view.append(img)
        #view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]

    def _aug_frame(
        self,
        buffer,
        args,
    ):
        h, w, _ = buffer[0].shape
        # Perform data augmentation - padding
        do_pad = video_transforms.pad_wide_clips(h, w, args.input_size)
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

    def _getitem_finetune_align(self, index):
        sample = self.dataset_samples[index]
        if self.video_loader:
            buffer, _, __ = self.load_images_cv2(sample, short_size=self.short_size)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images_cv2(sample, short_size=self.short_size)

        buffer = self._aug_frame(buffer, args)
        process_data, mask = self.transform((buffer, None))  # T*C,H,W
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data = process_data.view((self.view_len, 3) + process_data.size()[-2:]).transpose(0, 1)
        return (process_data, mask)
    
    def _getitem_orig(self, index):
        sample = self.dataset_samples[index]
        if self.video_loader:
            buffer, _, __ = self.load_images(sample, short_size=self.short_size)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample, short_size=320)

        process_data, mask = self.transform((buffer, None))  # T*C,H,W
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data = process_data.view((self.view_len, 3) + process_data.size()[-2:]).transpose(0, 1)
        return (process_data, mask)

    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return len(self.dataset_samples)



class MockArgs:
    def __init__(self):
        self.input_size = 224  # Example input size
        self.mask_type = 'tube'  # Masking type, 'tube' in this case
        self.window_size = (8, 14, 14)  # Example window size for TubeMaskingGenerator
        self.mask_ratio = 0.90  # Example mask ratio
        self.transforms_finetune_align = True
        self.aa = 'rand-m3-n3-mstd0.5-inc1' # "rand-m7-n4-mstd0.5-inc1"
        self.train_interpolation = "bicubic"


if __name__ == "__main__":
    from datasets_frame import DataAugmentationForVideoMAE, DataAugmentationForVideoMAE_LightCrop
    args = MockArgs()
    tf = DataAugmentationForVideoMAE_LightCrop(args)

    dataset = FrameClsDataset_DoTA(
        anno_path='train_split.txt',
        data_path="/datasets/DoTA",
        video_ext='mp4',
        is_color=True,
        view_len=16,
        view_step=1,
        orig_fps=10,
        target_fps=10,
        transform=tf,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        args=args,
    )
    L = len(dataset)
    labels = dataset._label_array
    assert len(labels) == L, f"L={L}, labels len is {len(labels)}"
    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)

    print(f"Dataset length: {L} for view step {dataset.view_step}")
    print("unique values and their counts:")
    for u, c in zip(unique, counts):
        print(f"item {u}: {c} times")

    item = dataset[0]
    #print("\nitem 0: \n", item)

