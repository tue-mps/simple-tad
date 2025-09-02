import os
import zipfile
import cv2
import numpy as np
import torch
import pandas as pd
import json
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from natsort import natsorted

from functional import crop_clip
from random_erasing import RandomErasing
import warnings
from torch.utils.data import Dataset
import video_transforms as video_transforms 
import volume_transforms as volume_transforms

from bdd100k import VideoMAE_BDD100K
from dataset.sequencing import RegularSequencer, UnsafeOverlapSequencer, RegularSequencerWithStart
from dataset.data_utils import smooth_labels, compute_time_vector


class FrameClsDataset_DADA(Dataset):
    """Load your own video classification dataset."""
    ego_categories = [str(cat) for cat in list(range(1, 19)) + [61, 62]]

    def __init__(self, anno_path, data_path, mode='train',
                 view_len=8, target_fps=10, orig_fps=30, view_step=10,
                 crop_size=224, short_side_size=320, video_ext=".png",
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
        self.video_ext = video_ext
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
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(self.crop_size, self.crop_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = [(0, 0)]
            self.test_dataset = self.dataset_samples
            self.test_label_array = self.label_array

    def _read_anno(self):
        clip_timesteps = []
        clip_binary_labels = []
        clip_cat_labels = []
        clip_ego = []
        clip_night = []
        clip_toa = []
        clip_ttc = []
        clip_acc = []
        clip_smoothed_labels = []

        errors = []

        with open(os.path.join(self.data_path, self.anno_path), 'r') as file:
            clip_names = [line.rstrip() for line in file]

        df = pd.read_csv(os.path.join(self.data_path, "annotation", "full_anno.csv"))

        for clip in tqdm(clip_names, "Part 1/2. Reading and checking clips"):
            clip_type, clip_subfolder = clip.split("/")
            row = df[(df["video"] == int(clip_subfolder)) & (df["type"] == int(clip_type))]
            info = f"clip: {clip}, type: {clip_type}, subfolder: {clip_subfolder}, rows found: {row}"
            description_csv = row["texts"]
            assert len(row) == 1, f"Multiple results! \n{info}"
            if len(row) != 1:
                errors.append(info)
            row = row.iloc[0]
            with zipfile.ZipFile(os.path.join(self.data_path, "frames", clip, "images.zip"), 'r') as zipf:
                framenames = natsorted([f for f in zipf.namelist() if os.path.splitext(f)[1]==self.video_ext])
            timesteps = natsorted([int(os.path.splitext(f)[0].split("_")[-1]) for f in framenames])
            if_acc_video = int(row["whether an accident occurred (1/0)"])
            st = int(row["abnormal start frame"])
            en = int(row["abnormal end frame"])
            if st > -1 and en > -1:
                binary_labels = [1 if st <= t <= en else 0 for t in timesteps]
            else:
                binary_labels = [0 for t in timesteps]
            cat_labels = [l*int(clip_type) for l in binary_labels]
            if_ego = clip_type in self.ego_categories
            if_night = int(row["light(day,night)1-2"]) == 2
            toa = int(row["accident frame"])
            ttc = compute_time_vector(binary_labels, fps=self.orig_fps, TT=self.ttc_TT, TA=self.ttc_TA)
            smoothed_labels = smooth_labels(labels=torch.Tensor(binary_labels), time_vector=ttc,
                                            before_limit=self.ttc_TT, after_limit=self.ttc_TA)

            clip_timesteps.append(timesteps)
            clip_binary_labels.append(binary_labels)
            clip_cat_labels.append(cat_labels)
            clip_ego.append(if_ego)
            clip_night.append(if_night)
            clip_toa.append(toa)
            clip_ttc.append(ttc)
            clip_acc.append(if_acc_video)
            clip_smoothed_labels.append(smoothed_labels)

        for line in errors:
            print(line)
        if len(errors) > 0:
            print(f"\n====\nerrors: {len(errors)}. You can add saving the error list in the code.")
            exit(0)

        assert len(clip_names) == len(clip_timesteps) == len(clip_binary_labels) == len(clip_cat_labels)
        self.clip_names = clip_names
        self.clip_timesteps = clip_timesteps
        self.clip_bin_labels = clip_binary_labels
        self.clip_cat_labels = clip_cat_labels
        self.clip_ego = clip_ego
        self.clip_night = clip_night
        self.clip_toa = clip_toa
        self.clip_ttc = clip_ttc
        self.clip_smoothed_labels = clip_smoothed_labels

    def _prepare_views(self):
        dataset_sequences = []
        label_array = []
        ttc = []
        smoothed_label_array = []
        sequencer = RegularSequencer(seq_frequency=self.target_fps, seq_length=self.view_len, step=self.view_step)
        N = len(self.clip_names)
        for i in tqdm(range(N), desc="Part 2/2. Preparing views"):
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
            buffer, _, __ = self.load_images_zip(sample, final_resize=False, resize_scale=1.)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images_zip(sample, final_resize=False, resize_scale=1.)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                smoothed_label_list = []
                index_list = []
                ttc_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    smoothed_label = self._smoothed_label_array[index]
                    ttc = self.ttc[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    smoothed_label_list.append(smoothed_label)
                    index_list.append(index)
                    ttc_list.append(ttc)
                extra_info = [{"ttc": ttc_item, "smoothed_labels": slab_item} for ttc_item, slab_item in
                              zip(ttc_list, smoothed_label_list)]
                return frame_list, label_list, index_list, extra_info
            else:
                buffer = self._aug_frame(buffer, args)
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index]}
            return buffer, self.label_array[index], index, extra_info

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer, _, __ = self.load_images_zip(sample, final_resize=True)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images_zip(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            extra_info = {"ttc": self.ttc[index], "smoothed_labels": self._smoothed_label_array[index]}
            return buffer, self.label_array[index], index, extra_info

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            buffer, clip_name, frame_name = self.load_images_zip(sample, final_resize=True)
            while len(buffer) == 0:
                warnings.warn("video {} not found during testing".format(str(self.test_dataset[index])))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                buffer, clip_name, frame_name = self.load_images_zip(sample, final_resize=True)
            buffer = self.data_transform(buffer)
            extra_info = {"ttc": self.ttc[index], "clip": clip_name, "frame": frame_name,
                          "smoothed_labels": self._smoothed_label_array[index]}
            return buffer, self.test_label_array[index], index, extra_info
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
            buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return buffer

    def load_images(self, dataset_sample, final_resize=False, resize_scale=None):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        subclip = clip_name.split("/")[1]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{subclip}_frame_{ts}{self.video_ext}" for ts in timesteps]
        view = []
        for fname in filenames:
            img = cv2.imread(os.path.join(self.data_path, "frames", clip_name, fname))
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

    def load_images_zip(self, dataset_sample, final_resize=False, resize_scale=None):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clip_names[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(4)}{self.video_ext}" for ts in timesteps]
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
                    short_side = min(img.shape[:2])
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


class VideoMAE_DADA2K(VideoMAE_BDD100K):
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
    ego_categories = [str(cat) for cat in list(range(1, 19)) + [61, 62]]

    def __init__(self, **kwargs):
        super(VideoMAE_DADA2K, self).__init__(**kwargs)

    def _make_dataset_snellius(self, directory, setting):
        clip_timesteps = []
        with open(os.path.join(self.root, self.setting), 'r') as file:
            clip_names = [line.rstrip() for line in file]
        df = pd.read_csv(os.path.join(self.root, "annotation", "full_anno.csv"))
        for clip in tqdm(clip_names, desc=" 1 | Gathering and checking clips"):
            clip_type, clip_subfolder = clip.split("/")
            row = df[(df["video"] == int(clip_subfolder)) & (df["type"] == int(clip_type))]
            info = f"clip: {clip}, type: {clip_type}, subfolder: {clip_subfolder}, rows found: {row}"
            description_csv = row["texts"]
            assert len(row) == 1, f"Multiple results! \n{info}"
            row = row.iloc[0]
            with zipfile.ZipFile(os.path.join(self.root, "frames", clip, "images.zip"), 'r') as zipf:
                framenames = natsorted([f for f in zipf.namelist() if os.path.splitext(f)[1]==self.video_ext])
            try:
                timesteps = natsorted([int(os.path.splitext(f)[0].split("_")[-1]) for f in framenames])
            except ValueError:
                print(f"ERR: {clip}")
                continue
            clip_timesteps.append(timesteps)

        assert len(clip_names) == len(clip_timesteps)
        #self.clips = clip_names
        self.clip_timesteps = clip_timesteps
        return clip_names

    def _prepare_views(self):
        dataset_sequences = []
        N = len(self.clips)
        for i in tqdm(range(N), desc=" 2 | Preparing views"):
            timesteps = self.clip_timesteps[i]
            sequences = self.sequencer.get_sequences(timesteps_nb=len(timesteps), input_frequency=self.fps)
            if sequences is None:
                continue
            dataset_sequences.extend([(i, seq) for seq in sequences])
        self.dataset_samples = dataset_sequences
    
    def load_images(self, dataset_sample):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clips[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(6)}{self.video_ext}" for ts in timesteps]
        view = []
        with zipfile.ZipFile(os.path.join(self.root, "frames", clip_name, "images.zip"), 'r') as zipf:
            for fname in filenames:
                with zipf.open(fname) as file:
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    print("Image doesn't exist! ", fname)
                    exit(1)
                # resze
                if self.intermediate_size is not None:
                    h, w, _ = img.shape
                    short_size = min([h, w, self.intermediate_size])
                    if h < w:
                        scale = short_size / h
                        new_h, new_w = short_size, int(w * scale)
                    else:
                        scale = short_size / w
                        new_h, new_w = int(h * scale), short_size
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # Convert OpenCV image (numpy) to PIL.Image and append to view
                img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                view.append(img)
        #view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]
    
    def load_images_cv2(self, dataset_sample):
        clip_id, frame_seq = dataset_sample
        clip_name = self.clips[clip_id]
        timesteps = [self.clip_timesteps[clip_id][idx] for idx in frame_seq]
        filenames = [f"{str(ts).zfill(6)}{self.video_ext}" for ts in timesteps]
        view = []
        try:
            with zipfile.ZipFile(os.path.join(self.root, "frames", clip_name, "images.zip"), 'r') as zipf:
                for fname in filenames:
                    with zipf.open(fname) as file:
                        file_bytes = np.frombuffer(file.read(), np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    if img is None:
                        print("Image doesn't exist! ", fname)
                        exit(1)
                    # resze
                    if self.intermediate_size is not None:
                        h, w, _ = img.shape
                        short_size = min([h, w, self.intermediate_size])
                        if h < w:
                            scale = short_size / h
                            new_h, new_w = short_size, int(w * scale)
                        else:
                            scale = short_size / w
                            new_h, new_w = int(h * scale), short_size
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    view.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"\nERROR file {os.path.join(self.root, 'frames', clip_name, 'images.zip')}")
            print(e)
            
        #view = np.stack(view, axis=0)
        return view, clip_name, filenames[-1]
    
    # def _aug_frame(
    #     self,
    #     buffer,
    #     args,
    # ):
    #     if torch.rand(1).item() > 0.3:
    #         h, w, _ = buffer[0].shape
    #         # Perform data augmentation - padding
    #         do_pad = video_transforms.pad_wide_clips(h, w, args.input_size)
    #         buffer = [do_pad(img) for img in buffer]

    #         aug_transform = video_transforms.create_random_augment(
    #             input_size=(args.input_size, args.input_size),
    #             auto_augment=args.aa,
    #             interpolation=args.train_interpolation,
    #             do_transforms=video_transforms.DRIVE_TRANSFORMS
    #         )

    #         buffer = [transforms.ToPILImage()(frame) for frame in buffer]
    #         buffer = aug_transform(buffer)

    #     return buffer
    
    def _getitem_finetune_align(self, index):
        sample = self.dataset_samples[index]
        if self.video_loader:
            buffer, _, __ = self.load_images_cv2(sample)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images_cv2(sample)

        buffer = self._aug_frame(buffer, self.args)
        process_data, mask = self.transform((buffer, None))  # T*C,H,W
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0, 1)
        return (process_data, mask)

    def _getitem_orig(self, index):
        sample = self.dataset_samples[index]
        if self.video_loader:
            buffer, _, __ = self.load_images(sample)  # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer, _, __ = self.load_images(sample)

        process_data, mask = self.transform((buffer, None))  # T*C,H,W
        # T*C,H,W -> T,C,H,W -> C,T,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0, 1)
        return (process_data, mask)

    def __len__(self):
        return len(self.dataset_samples)
    

class VideoMAE_DADA2K_prepared(VideoMAE_DADA2K):

    def __init__(self, clips_txt, timesteps_pkl, views_pkl, **kwargs):
        self.clips_txt = clips_txt
        self.timesteps_pkl = timesteps_pkl
        self.views_pkl = views_pkl
        super().__init__(**kwargs)

    def _make_dataset_snellius(self, directory, setting):
        clips = []
        # read from the file
        with open(self.clips_txt, 'r') as file:
            clips = [line.rstrip() for line in file]
        assert len(clips) > 0, f"Cannot find any video clips for the given split: {setting}"
        with open(self.timesteps_pkl, 'rb') as file:
            timesteps = pickle.load(file)
            assert len(timesteps) == len(clips)
        self.clip_timesteps = timesteps
        return clips
    
    def _prepare_views(self):
        dataset_sequences = []
        # read from the file
        with open(self.views_pkl, 'rb') as file:
            dataset_sequences = pickle.load(file)
        self.dataset_samples = dataset_sequences


class MockArgs:
    def __init__(self):
        self.input_size = 224  # Example input size
        self.mask_type = 'tube'  # Masking type, 'tube' in this case
        self.window_size = (8, 14, 14)  # Example window size for TubeMaskingGenerator
        self.mask_ratio = 0.90  # Example mask ratio
        self.loss = "crossentropy"
        self.transforms_finetune_align = True
        self.reprob = 0.
        self.aa = "rand-m3-n3-mstd0.5-inc1"
        self.train_interpolation = "bicubic"


if __name__ == "__main__":
    from datasets import DataAugmentationForVideoMAE
    args = MockArgs()
    tf = DataAugmentationForVideoMAE(args)

    if False:
        dataset = FrameClsDataset_DADA(
                anno_path="DADA2K_my_split/half_training.txt",
                data_path="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/DADA2K",
                mode="train",
                view_len=16,
                view_step=1,
                orig_fps=30,  # original FPS of the dataset
                target_fps=10,  # 10
                num_segment=1,
                test_num_segment=1,
                test_num_crop=1,  # 1
                num_crop=1,
                keep_aspect_ratio=True,
                crop_size=args.input_size,
                args=args)
        print(f"\nDATASET views: {len(dataset)}")
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
        exit(0)

    if False:
        dataset = VideoMAE_DADA2K(
            root="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/DADA2K",
            setting="DADA2K_my_split/all.txt",
            video_ext='.png',
            is_color=True,
            new_length=16,
            new_step=1,
            fps=30,
            target_fps=10,
            transform=tf,
            temporal_jitter=False,
            video_loader=True,
            use_decord=True,
            lazy_init=False,
            args=args
        )
        L = len(dataset)
        print(f"Dataset length: {L} for view step {dataset.new_step}")

        item = dataset[0]
        #print("\nitem 0: \n", item)

    if True:
        dataset = VideoMAE_DADA2K_prepared(
        clips_txt="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/prepared_splits/training_clips.txt",
        timesteps_pkl="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/prepared_splits/training_timesteps.pkl",
        views_pkl="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/prepared_splits/training_dataset_samples.pkl",
        root="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA",
        setting="CAPDATA_my_split/training.txt",
        video_ext='.jpg',
        is_color=True,
        new_length=16,
        new_step=1,
        fps=30,
        target_fps=10,
        transform=tf,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        args=args
        )
        L = len(dataset)
        print(f"Dataset length: {L} for view step {dataset.new_step}")
        print("Sample example:")
        print(dataset.dataset_samples[0])
        exit(0)


    print("\n=======================\nCAP-DATA\n========================")

    dataset = VideoMAE_DADA2K(
        root="/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA",
        setting="CAPDATA_my_split/training.txt",
        video_ext='.jpg',
        is_color=True,
        new_length=16,
        new_step=1,
        fps=30,
        target_fps=10,
        transform=tf,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        args=args
    )
    L = len(dataset)
    print(f"Dataset length: {L} for view step {dataset.new_step}")

    item = dataset[0]
    # print("\nitem 0: \n", item)

    assert len(dataset.clips) == len(dataset.clip_timesteps)

    print("clip timesteps:")
    print(dataset.clip_timesteps[:5])

    os.makedirs("/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/prepared_splits", exist_ok=False)  # Don't want to rewrite

    print("Writing clips...")
    with open("/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/prepared_splits/training_clips.txt", "w") as f:
        for c in dataset.clips:
            f.write(c + "\n")
    print("Done!")

    print("Writing timesteps...")
    with open("/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/prepared_splits/training_timesteps.pkl", "wb") as f:
        pickle.dump(dataset.clip_timesteps, f)
    print("Done!")

    print("Writing views...")

    with open("/gpfs/work3/0/tese0625/RiskNetData/LOTVS-DADA/CAP-DATA/prepared_splits/training_dataset_samples.pkl", "wb") as f:
        pickle.dump(dataset.dataset_samples, f)

    print("Done!")

    exit(0)
