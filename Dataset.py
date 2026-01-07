import random
import torch
import numpy as np
import cv2
import glob
import os
import scipy.io as scio
from torch.utils.data import Dataset


def np_load_frame(filename, resize_h, resize_w):
    img = cv2.imread(filename)
    if img is None:
        raise ValueError(f"Failed to read image: {filename}")

    image_resized = cv2.resize(img, (resize_w, resize_h)).astype('float32')
    image_resized = (image_resized / 127.5) - 1.0  # [-1, 1]
    image_resized = np.transpose(image_resized, [2, 0, 1])  # (C, H, W)
    return image_resized


class train_dataset(Dataset):
    """
    Training dataset (normal videos only).
    No data augmentation.
    """

    def __init__(self, cfg):
        self.img_h = cfg.img_size[0]
        self.img_w = cfg.img_size[1]
        self.clip_length = 5

        self.videos = []
        self.all_seqs = []

        # 🔥 Ensure deterministic ordering of folders
        video_folders = sorted(glob.glob(os.path.join(cfg.train_data, '*')))

        for folder in video_folders:
            # 🔥 Ensure deterministic ordering of frames
            all_imgs = sorted(glob.glob(os.path.join(folder, '*.jpg')))

            if len(all_imgs) < self.clip_length:
                continue

            self.videos.append(all_imgs)

            # valid starting indices
            seq_indices = list(range(len(all_imgs) - self.clip_length + 1))
            random.shuffle(seq_indices)
            self.all_seqs.append(seq_indices)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, indice):
        one_folder = self.videos[indice]

        video_clip = []
        start = self.all_seqs[indice][-1]  # use last shuffled index

        for i in range(start, start + self.clip_length):
            video_clip.append(
                np_load_frame(one_folder[i], self.img_h, self.img_w)
            )

        video_clip = np.array(video_clip).reshape((-1, self.img_h, self.img_w))
        video_clip = torch.from_numpy(video_clip)

        # flow index string (used by FlowNet)
        flow_str = f'{indice}_{start + 3}-{start + 4}'
        return indice, video_clip, flow_str


class test_dataset:
    """
    Test dataset (no labels).
    Produces sliding clips of length = clip_length.
    """

    def __init__(self, cfg, video_folder):
        self.img_h = cfg.img_size[0]
        self.img_w = cfg.img_size[1]
        self.clip_length = 5

        # 🔥 Strict ordering
        self.imgs = sorted(glob.glob(os.path.join(video_folder, '*.jpg')))

        if len(self.imgs) < self.clip_length:
            raise ValueError(f"Not enough frames in {video_folder}")

    def __len__(self):
        return len(self.imgs) - (self.clip_length - 1)

    def __getitem__(self, indice):
        video_clips = []

        for frame_id in range(indice, indice + self.clip_length):
            video_clips.append(
                np_load_frame(self.imgs[frame_id], self.img_h, self.img_w)
            )

        video_clips = np.array(video_clips).reshape((-1, self.img_h, self.img_w))
        return video_clips


class Label_loader:
    """
    Only used for evaluation on official datasets.
    NOT used during Kaggle inference.
    """

    def __init__(self, cfg, video_folders):
        assert cfg.dataset in ('ped2', 'avenue', 'shanghaitech'), \
            f"Unknown dataset: {cfg.dataset}"

        self.cfg = cfg
        self.name = cfg.dataset
        self.frame_path = cfg.test_data
        self.mat_path = os.path.join(cfg.data_root, self.name, f'{self.name}.mat')
        self.video_folders = video_folders

    def __call__(self):
        if self.name == 'shanghaitech':
            return self.load_shanghaitech()
        else:
            return self.load_ucsd_avenue()

    def load_ucsd_avenue(self):
        abnormal_events = scio.loadmat(self.mat_path, squeeze_me=True)['gt']

        all_gt = []
        for i in range(abnormal_events.shape[0]):
            # 🔥 Ensure frame count matches actual frames
            frames = sorted(os.listdir(self.video_folders[i]))
            length = len(frames)

            sub_video_gt = np.zeros((length,), dtype=np.int8)
            one_abnormal = abnormal_events[i]

            if one_abnormal.ndim == 1:
                one_abnormal = one_abnormal.reshape((one_abnormal.shape[0], -1))

            for j in range(one_abnormal.shape[1]):
                start = int(one_abnormal[0, j]) - 1
                end = int(one_abnormal[1, j])
                sub_video_gt[start:end] = 1

            all_gt.append(sub_video_gt)

        return all_gt

    def load_shanghaitech(self):
        np_list = sorted(
            glob.glob(os.path.join(self.cfg.data_root, self.name, 'frame_masks', '*.npy'))
        )

        gt = [np.load(npy) for npy in np_list]
        return gt
