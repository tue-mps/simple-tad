import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
#from numpy.lib.function_base import disp
import torch
import decord
import random
from PIL import Image
from torchvision import transforms
from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset, DataLoader
import video_transforms as video_transforms 
import volume_transforms as volume_transforms


kinetics_700_ignore_list = (
    "z3G3aq7olkE", "qUPijtzm3j0", "y4OSrkGHTU0", "WdfC1Oq4PqY", "5d9mIpws4cg", 
    "A-FCzUzEd4U", "y7cYaYX4gdw", "SYTMgaqGhfg", "BSN_nDiTwBo", "zLD_q2djrYs", 
    "NNazT7dDWxA", "_dbw-EJqoMY", "ixQrfusr6k8", "FAqHwAPZfeE"
)


class VideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(self, anno_path, data_path, mode='train', clip_len=8,
                 frame_sample_rate=2, crop_size=224, short_side_size=256,
                 new_height=256, new_width=340, keep_aspect_ratio=True,
                 num_segment=1, num_crop=1, test_num_segment=10, test_num_crop=3,args=None):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if self.mode in ['train']:
            self.aug = True
            if self.args.reprob > 0:
                self.rand_erase = True
        if VideoReader is None:
            raise ImportError("Unable to import `decord` which is required to read videos.")

        import pandas as pd
        #cleaned = pd.read_csv(self.anno_path, header=None, delimiter=' ')
        cleaned = pd.read_csv(self.anno_path, header=None)
        self.dataset_samples = list(cleaned.values[:, 0])
        self.label_array = list(cleaned.values[:, 1])

        if (mode == 'train'):
            pass

        elif (mode == 'validation'):
            self.data_transform = video_transforms.Compose([
                video_transforms.Resize(self.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
        elif mode == 'test':
            self.data_resize = video_transforms.Compose([
                video_transforms.Resize(size=(short_side_size), interpolation='bilinear')
            ])
            self.data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
            ])
            self.test_seg = []
            self.test_dataset = []
            self.test_label_array = []
            for ck in range(self.test_num_segment):
                for cp in range(self.test_num_crop):
                    for idx in range(len(self.label_array)):
                        sample_label = self.label_array[idx]
                        self.test_label_array.append(sample_label)
                        self.test_dataset.append(self.dataset_samples[idx])
                        self.test_seg.append((ck, cp))

    def __getitem__(self, index):
        if self.mode == 'train':
            args = self.args 
            scale_t = 1

            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t) # T H W C
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during training".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample, sample_rate_scale=scale_t)

            if args.num_sample > 1:
                frame_list = []
                label_list = []
                index_list = []
                for _ in range(args.num_sample):
                    new_frames = self._aug_frame(buffer, args)
                    label = self.label_array[index]
                    frame_list.append(new_frames)
                    label_list.append(label)
                    index_list.append(index)
                return frame_list, label_list, index_list, {}
            else:
                buffer = self._aug_frame(buffer, args)
            
            return buffer, self.label_array[index], index, {}

        elif self.mode == 'validation':
            sample = self.dataset_samples[index]
            buffer = self.loadvideo_decord(sample)
            if len(buffer) == 0:
                while len(buffer) == 0:
                    warnings.warn("video {} not correctly loaded during validation".format(sample))
                    index = np.random.randint(self.__len__())
                    sample = self.dataset_samples[index]
                    buffer = self.loadvideo_decord(sample)
            buffer = self.data_transform(buffer)
            return buffer, self.label_array[index], sample.split("/")[-1].split(".")[0]

        elif self.mode == 'test':
            sample = self.test_dataset[index]
            chunk_nb, split_nb = self.test_seg[index]
            buffer = self.loadvideo_decord(sample)

            while len(buffer) == 0:
                warnings.warn("video {}, temporal {}, spatial {} not found during testing".format(\
                    str(self.test_dataset[index]), chunk_nb, split_nb))
                index = np.random.randint(self.__len__())
                sample = self.test_dataset[index]
                chunk_nb, split_nb = self.test_seg[index]
                buffer = self.loadvideo_decord(sample)

            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            spatial_step = 1.0 * (max(buffer.shape[1], buffer.shape[2]) - self.short_side_size) \
                                 / (self.test_num_crop - 1)
            temporal_step = max(1.0 * (buffer.shape[0] - self.clip_len) \
                                / (self.test_num_segment - 1), 0)
            temporal_start = int(chunk_nb * temporal_step)
            spatial_start = int(split_nb * spatial_step)
            if buffer.shape[1] >= buffer.shape[2]:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       spatial_start:spatial_start + self.short_side_size, :, :]
            else:
                buffer = buffer[temporal_start:temporal_start + self.clip_len, \
                       :, spatial_start:spatial_start + self.short_side_size, :]

            buffer = self.data_transform(buffer)
            return buffer, self.test_label_array[index], sample.split("/")[-1].split(".")[0], \
                   chunk_nb, split_nb
        else:
            raise NameError('mode {} unkown'.format(self.mode))

    def _aug_frame(
        self,
        buffer,
        args,
    ):

        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )

        buffer = [
            transforms.ToPILImage()(frame) for frame in buffer
        ]

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
        # Perform data augmentation.
        scl, asp = (
            [0.08, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=False if args.data_set == 'SSV2' else True ,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False
        )

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


    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print('SKIP: ', fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(fname, width=self.new_width, height=self.new_height,
                                 num_threads=1, ctx=cpu(0))
        except:
            print("video cannot be loaded by decord: ", fname)
            return []

        if self.mode == 'test':
            all_index = [x for x in range(0, len(vr), self.frame_sample_rate)]
            while len(all_index) < self.clip_len:
                all_index.append(all_index[-1])
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = len(vr) // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate((index, np.ones(self.clip_len - seg_len // self.frame_sample_rate) * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = np.random.randint(converted_len, seg_len)
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i*seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer

    def __len__(self):
        if self.mode != 'test':
            return len(self.dataset_samples)
        else:
            return len(self.test_dataset)


def sample_frame_window(video_reader, new_length, target_fps):
    # 1) get and round source fps
    src_fps = int(round(video_reader.get_avg_fps()))
    assert abs(src_fps - target_fps) > 1, \
        f"Source FPS ({src_fps}) not divisible by target {target_fps}"
    step = int(src_fps // target_fps)   # e.g. 30//10 == 3

    # 2) how many raw frames we need in one window
    window_size = new_length * step

    # 3) video length in frames
    n = len(video_reader)
    if n < window_size:
        raise ValueError(f"Video too short ({n} frames) for window {window_size}")

    # 4) pick a random start so the whole window fits
    start = random.randint(0, n - window_size)

    # 5) build your target indices: start, start+step, start+2*step, …​
    return [start + i*step for i in range(new_length)]


def sample_interpolated_window(vr: VideoReader,
                               new_length: int,
                               target_fps: float):
    """
    Return `new_length` frames sampled at `target_fps` by linear interpolation
    from whatever the source FPS is.
    """
    # 0) sync Python RNG to torch, if you want reproducibility
    seed32 = torch.initial_seed() & 0xFFFFFFFF
    random.seed(seed32)
    
    # 1) source stats
    src_fps = vr.get_avg_fps()
    n_frames = len(vr)
    duration = n_frames / src_fps
    window_dur = (new_length - 1) / target_fps
    
    if duration < window_dur:
        raise ValueError(f"Video too short ({duration:.2f}s) for "
                         f"{new_length} @ {target_fps}FPS "
                         f"→ needs ≥{window_dur:.2f}s")
    
    # 2) pick a random start time so the window fits
    t0 = random.random() * (duration - window_dur)
    
    # 3) desired timestamps (s) and float‐indices
    ts = t0 + np.arange(new_length) / target_fps        # shape [L]
    f_idx = ts * src_fps                                # shape [L]
    
    # 4) floor/ceil indices & interpolation weights
    i0 = np.floor(f_idx).astype(int)                    # shape [L]
    i1 = np.minimum(i0 + 1, n_frames - 1)               # shape [L]
    a  = (f_idx - i0).astype(np.float32)                # shape [L]
    
    # 5) fetch all needed frames in one call
    #    interleave so we only do one get_batch
    full_idxs = np.stack([i0, i1], axis=1).reshape(-1)  # shape [2L]
    try:
        raw = vr.get_batch(full_idxs.tolist())              # decord NDArray: [2L,H,W,C]
    except:
        raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, video_name, duration))
    raw = raw.asnumpy().astype(np.float32)              # → numpy [2L,H,W,C]
    
    # 6) split and blend
    floor_frames = raw[0::2]
    ceil_frames  = raw[1::2]
    a = a[:, None, None, None]                          # [L,1,1,1]
    out = (1 - a) * floor_frames + a * ceil_frames      # [L,H,W,C]
    
    # 7) back to uint8
    return out.clip(0,255).astype(np.uint8)


def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
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


class VideoMAE(torch.utils.data.Dataset):
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
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 intermediate_size=320,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False,
                 manager=None,
                 args=None,
                 prepared_clips_file=None):

        super(VideoMAE, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.intermediate_size = intermediate_size
        self.transform = transform
        self.lazy_init = lazy_init
        self.args = args

        if not self.lazy_init:
            self.clips = self._make_dataset_snellius(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))
            
        # if prepared_clips_file:
        #     clips = []
        #     # read from the file, but lines are tuples!
        #     with open(prepared_clips_file, 'r') as file:
        #         clips = [line.rstrip() for line in file]
        #     assert len(clips) > 0, f"Cannot find any video clips for the given split: {setting}"
        #     self.clips = clips
        
        if manager:
            self.corrupt_clips = manager.list()
            self.corrupt_clips_decord = manager.list()
    
    def __getitem__(self, index):

        directory, target = self.clips[index]
        if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
        else:
            # data in the "setting" file do not have extension, e.g., demo
            # So we need to provide extension (i.e., .mp4) to complete the file name.
            video_name = '{}.{}'.format(directory, self.video_ext)

        if self.video_loader:
            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
            segment_indices, skip_offsets = self._sample_train_indices(duration)
            images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
            assert len(images) > 0

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        
        return (process_data, mask)
    
    def __getitem__check(self, index):

        directory, target = self.clips[index]
        if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
        else:
            # data in the "setting" file do not have extension, e.g., demo
            # So we need to provide extension (i.e., .mp4) to complete the file name.
            video_name = '{}.{}'.format(directory, self.video_ext)

        #if self.video_loader:
        try:
            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
            segment_indices, skip_offsets = self._sample_train_indices(duration)
            images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
            assert len(images) > 0
            return 1
        except Exception as e:
            self.corrupt_clips_decord.append(video_name)
            print(f"Decord failed for {video_name} with error: {e}. Falling back to OpenCV.")
            # Fall back to OpenCV
            cap = cv2.VideoCapture(video_name)
            if not cap.isOpened():
                #raise RuntimeError(f"Error: Unable to open video file {video_name}")
                self.corrupt_clips.append(video_name)
                return 0
            duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if duration < self.new_length:
                self.corrupt_clips.append(video_name)
            return 0
            segment_indices, skip_offsets = self._sample_train_indices(duration)
            images = self._video_TSN_opencv_batch_loader(cap, duration, segment_indices, skip_offsets)
            assert len(images) > 0

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        
        return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
        return clips
    
    def _make_dataset_snellius(self, directory, setting):
        subset = os.path.splitext(os.path.basename(setting))[0]
        if not os.path.exists(os.path.join(directory, subset)):
            raise RuntimeError(f"Subset directory does not exist! {os.path.join(directory, subset)}")
        if not os.path.exists(os.path.join(directory, setting)):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (os.path.join(directory, setting))))
        clips = []
        df = pd.read_csv(os.path.join(directory, setting))
        # remove corrupted or too short clips
        df = df[~df['youtube_id'].isin(kinetics_700_ignore_list)]
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            target = row["label"]
            ytid = row["youtube_id"]
            t1 = str(int(row["time_start"])).zfill(6)
            t2 = str(int(row["time_end"])).zfill(6)
            clip_path = os.path.join(directory, subset, target, f"{ytid}_{t1}_{t2}.mp4")
            if not os.path.exists(clip_path):
                raise RuntimeError(f"Video does not exist! {clip_path}")
            clips.append((clip_path, target))
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def decord_extract_frames(self, video_reader, frame_id_list, duration=None, video_name=None):
        sampled_list = []
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            # Resize frames while maintaining aspect ratio
            resized_frames = []
            for frame in video_data:
                h, w, _ = frame.shape
                short_size = min([h, w, self.intermediate_size])
                if h < w:
                    scale = short_size / h
                    new_h, new_w = short_size, int(w * scale)
                else:
                    scale = short_size / w
                    new_h, new_w = int(h * scale), short_size

                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized_frame)
            sampled_list = [Image.fromarray(frame).convert('RGB') for frame in resized_frames]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, video_name, duration))
        return sampled_list
    
    def decord_extract_frames_cv2(self, video_reader, frame_id_list, duration=None, video_name=None):
        sampled_list = []
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            # Resize frames while maintaining aspect ratio
            for frame in video_data:
                h, w, _ = frame.shape
                if h < w:
                    scale = 320 / h
                    new_h, new_w = 320, int(w * scale)
                else:
                    scale = 320 / w
                    new_h, new_w = int(h * scale), 320

                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                sampled_list.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, video_name, duration))
        return sampled_list

    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            # Resize frames while maintaining aspect ratio
            resized_frames = []
            for frame in video_data:
                h, w, _ = frame.shape
                if h < w:
                    scale = 320 / h
                    new_h, new_w = 320, int(w * scale)
                else:
                    scale = 320 / w
                    new_h, new_w = int(h * scale), 320

                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized_frame)
            video_data = resized_frames
            sampled_list = [Image.fromarray(video_data[vid]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
            #sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list
    

    # DO NOT RECOMMEND - SLOW FOR VIDEOS!!!!! USE DECORD
    def _video_TSN_opencv_batch_loader(self, cap, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []

        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step

        for frame_id in frame_id_list:
            # Set the video frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            # Read the frame
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Error: Unable to read frame {frame_id} from video.")

            h, w, _ = frame.shape
            if h < w:
                scale = 320 / h
                new_h, new_w = 320, int(w * scale)
            else:
                scale = 320 / w
                new_h, new_w = int(h * scale), 320

            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Convert frame to RGB format (OpenCV reads in BGR by default)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Append the processed frame
            sampled_list.append(frame)

        return sampled_list
    

    def _video_TSN_opencv_batch_loader_debug(self, cap, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []

        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step

        max_id = max(frame_id_list)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            return 0
        else:
            return 1
    


class VideoMAE_aligned(torch.utils.data.Dataset):
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
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 intermediate_size=320,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False,
                 manager=None,
                 args=None,
                 target_fps=10):

        super(VideoMAE_aligned, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.intermediate_size = intermediate_size
        self.transform = transform
        self.lazy_init = lazy_init
        self.target_fps = target_fps
        self.args = args

        # make random use the same seed as pytorch
        seed64 = torch.initial_seed()
        seed32 = seed64 & 0xFFFFFFFF
        random.seed(seed32)

        if not self.lazy_init:
            self.clips = self._make_dataset_snellius(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))
            
        # if prepared_clips_file:
        #     clips = []
        #     # read from the file, but lines are tuples!
        #     with open(prepared_clips_file, 'r') as file:
        #         clips = [line.rstrip() for line in file]
        #     assert len(clips) > 0, f"Cannot find any video clips for the given split: {setting}"
        #     self.clips = clips
        
        if manager:
            self.corrupt_clips = manager.list()
            self.corrupt_clips_decord = manager.list()
    
    def __getitem__(self, index):

        directory, target = self.clips[index]
        if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
        else:
            # data in the "setting" file do not have extension, e.g., demo
            # So we need to provide extension (i.e., .mp4) to complete the file name.
            video_name = '{}.{}'.format(directory, self.video_ext)

        if self.video_loader:
            vr = decord.VideoReader(video_name, num_threads=1)
            # frame_seq = sample_frame_window(
            #     video_reader=vr,
            #     new_length=self.new_length,
            #     target_fps=self.target_fps
            # )
            video_data = sample_interpolated_window(
                vr=vr,
                new_length=self.new_length,
                target_fps=self.target_fps
            )
            #
            images = self.process_frames_cv2(video_data=video_data)
            assert len(images) == self.new_length, f"Video: {video_name}, sampled {len(images)} frames instead of {self.new_length}"

        # augment
        images = self._aug_frame(images, self.args)
        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0
                                                                                                    ,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        return (process_data, mask)

    def _aug_frame(self,buffer,args,):
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
    
    def __getitem__check(self, index):

        directory, target = self.clips[index]
        if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
        else:
            # data in the "setting" file do not have extension, e.g., demo
            # So we need to provide extension (i.e., .mp4) to complete the file name.
            video_name = '{}.{}'.format(directory, self.video_ext)

        #if self.video_loader:
        try:
            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
            segment_indices, skip_offsets = self._sample_train_indices(duration)
            images = self._video_TSN_decord_batch_loader(directory, decord_vr, duration, segment_indices, skip_offsets)
            assert len(images) > 0
            return 1
        except Exception as e:
            self.corrupt_clips_decord.append(video_name)
            print(f"Decord failed for {video_name} with error: {e}. Falling back to OpenCV.")
            # Fall back to OpenCV
            cap = cv2.VideoCapture(video_name)
            if not cap.isOpened():
                #raise RuntimeError(f"Error: Unable to open video file {video_name}")
                self.corrupt_clips.append(video_name)
                return 0
            duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if duration < self.new_length:
                self.corrupt_clips.append(video_name)
            return 0
            segment_indices, skip_offsets = self._sample_train_indices(duration)
            images = self._video_TSN_opencv_batch_loader(cap, duration, segment_indices, skip_offsets)
            assert len(images) > 0

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        
        return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
        return clips
    
    def _make_dataset_snellius(self, directory, setting):
        subset = os.path.splitext(os.path.basename(setting))[0]
        if not os.path.exists(os.path.join(directory, subset)):
            raise RuntimeError(f"Subset directory does not exist! {os.path.join(directory, subset)}")
        if not os.path.exists(os.path.join(directory, setting)):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (os.path.join(directory, setting))))
        clips = []
        #df = pd.read_csv(os.path.join(directory, setting))
        df = pd.read_csv("/projects/0/prjs1424/sveta/datasets/kinetics-700/train_long.csv")
        # remove corrupted or too short clips
        df = df[~df['youtube_id'].isin(kinetics_700_ignore_list)]
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):
            target = row["label"]
            ytid = row["youtube_id"]
            t1 = str(int(row["time_start"])).zfill(6)
            t2 = str(int(row["time_end"])).zfill(6)
            clip_path = os.path.join(directory, subset, target, f"{ytid}_{t1}_{t2}.mp4")
            if not os.path.exists(clip_path):
                raise RuntimeError(f"Video does not exist! {clip_path}")
            clips.append((clip_path, target))
        return clips

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def decord_extract_frames(self, video_reader, frame_id_list, duration=None, video_name=None):
        sampled_list = []
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            # Resize frames while maintaining aspect ratio
            resized_frames = []
            for frame in video_data:
                h, w, _ = frame.shape
                short_size = min([h, w, self.intermediate_size])
                if h < w:
                    scale = short_size / h
                    new_h, new_w = short_size, int(w * scale)
                else:
                    scale = short_size / w
                    new_h, new_w = int(h * scale), short_size

                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized_frame)
            sampled_list = [Image.fromarray(frame).convert('RGB') for frame in resized_frames]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, video_name, duration))
        return sampled_list
    
    def process_frames_cv2(self, video_data):
        sampled_list = []
        # Resize frames while maintaining aspect ratio
        for frame in video_data:
            h, w, _ = frame.shape
            if h < w:
                scale = 320 / h
                new_h, new_w = 320, int(w * scale)
            else:
                scale = 320 / w
                new_h, new_w = int(h * scale), 320

            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            sampled_list.append(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        return sampled_list

    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            # Resize frames while maintaining aspect ratio
            resized_frames = []
            for frame in video_data:
                h, w, _ = frame.shape
                if h < w:
                    scale = 320 / h
                    new_h, new_w = 320, int(w * scale)
                else:
                    scale = 320 / w
                    new_h, new_w = int(h * scale), 320

                resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                resized_frames.append(resized_frame)
            video_data = resized_frames
            sampled_list = [Image.fromarray(video_data[vid]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
            #sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list
    

    # DO NOT RECOMMEND - SLOW FOR VIDEOS!!!!! USE DECORD
    def _video_TSN_opencv_batch_loader(self, cap, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []

        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step

        for frame_id in frame_id_list:
            # Set the video frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            # Read the frame
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError(f"Error: Unable to read frame {frame_id} from video.")

            h, w, _ = frame.shape
            if h < w:
                scale = 320 / h
                new_h, new_w = 320, int(w * scale)
            else:
                scale = 320 / w
                new_h, new_w = int(h * scale), 320

            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Convert frame to RGB format (OpenCV reads in BGR by default)
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Append the processed frame
            sampled_list.append(frame)

        return sampled_list
    

    def _video_TSN_opencv_batch_loader_debug(self, cap, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []

        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step

        max_id = max(frame_id_list)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            return 0
        else:
            return 1
    


class MockArgs:
    def __init__(self):
        self.input_size = 224  # Example input size
        self.mask_type = 'tube'  # Masking type, 'tube' in this case
        self.window_size = (8, 14, 14)  # Example window size for TubeMaskingGenerator
        self.mask_ratio = 0.90  # Example mask ratio


class CustomDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize attributes to store corrupt clips
        self.corrupt_clips = []
        self.corrupt_clips_decord = []



if __name__ == "__main__":
    print("Start!")
    from datasets import DataAugmentationForVideoMAE
    from itertools import islice
    from multiprocessing import Manager
    manager = Manager()

    args = MockArgs()
    tf = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root='/scratch-nvme/ml-datasets/kinetics/k700-2020',
        setting="annotations/train.csv",
        prepared_clips_file="/gpfs/work3/0/tese0625/datasets/k700/prepared_clips.txt",
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=16,
        new_step=4,
        transform=tf,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        manager=None)
    L = len(dataset)
    
    print(f"\nDataset length: {L}, preparing samples...")

    clips = dataset.clips

    print("Writing clips...")
    with open("/gpfs/work3/0/tese0625/datasets/k700/prepared_clips.txt", "w") as file:
        for line in clips:
            file.write(line + "\n")
    print("\tClips done!")

    print("Done!")

    # for idx, batch in tqdm(enumerate(dataloader), total=L2, desc="Validating dataset"):
    #     _ = batch

    #     if idx % print_break == 0:
    #         problems = dataloader.dataset.corrupt_clips_decord
    #         problems_ = "\n".join(problems)
    #         if len(problems) > 0:
    #             with open(f"/home/sorlova/repos/AITHENA/NewStage/VideoMAE/scripts/kinetics2/decord_err_train_b200_{idx}.txt", mode="w") as f:
    #                 f.write(problems_)
    #         problems = dataloader.dataset.corrupt_clips
    #         problems_ = "\n".join(problems)
    #         if len(problems) > 0:
    #             with open(f"/home/sorlova/repos/AITHENA/NewStage/VideoMAE/scripts/kinetics2/opencv_err_train_b200_{idx}.txt", mode="w") as f:
    #                 f.write(problems_)
        
    # problems = dataloader.dataset.corrupt_clips_decord
    # problems_ = "\n".join(problems)
    # if len(problems) > 0:
    #     with open(f"/home/sorlova/repos/AITHENA/NewStage/VideoMAE/scripts/kinetics2/decord_err_train_b200.txt", mode="w") as f:
    #         f.write(problems_)
    # problems = dataloader.dataset.corrupt_clips
    # problems_ = "\n".join(problems)
    # if len(problems) > 0:
    #     with open(f"/home/sorlova/repos/AITHENA/NewStage/VideoMAE/scripts/kinetics2/opencv_err_train_b200.txt", mode="w") as f:
    #         f.write(problems_)

    # print("Done!")
    exit(0)

