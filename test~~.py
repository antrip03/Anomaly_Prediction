#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Dataset import test_dataset
from config import update_config
from models.unet import UNet


def rotate_180(x):
    # x: (B, C, H, W)
    return torch.rot90(x, 2, [2, 3])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='avenue')
    parser.add_argument('--trained_model', type=str, required=True)
    args = parser.parse_args()

    # -----------------------
    # Config
    # -----------------------
    cfg = update_config(args, mode='test')
    cfg.print_cfg()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -----------------------
    # Load model
    # -----------------------
    model = UNet(12)


    ckpt = torch.load(cfg.trained_model, map_location=device)

    # pretrained generator weights are stored under 'net_g'
    model.load_state_dict(ckpt['net_g'], strict=True)

    model.to(device)
    model.eval()

    # -----------------------
    # Prepare test videos
    # -----------------------
    video_folders = sorted([
        os.path.join(cfg.test_data, f)
        for f in os.listdir(cfg.test_data)
        if os.path.isdir(os.path.join(cfg.test_data, f))
    ])

    os.makedirs('results', exist_ok=True)

    # -----------------------
    # Inference
    # -----------------------
    for vid, video_folder in enumerate(video_folders, start=1):
        dataset = test_dataset(cfg, video_folder)
        scores = []

        for i in tqdm(range(len(dataset)), desc=f'Video {vid}'):
            clip = dataset[i]  # shape: (5*3, H, W)
            clip = torch.from_numpy(clip).unsqueeze(0).to(device)

            # Predict next frame (last frame supervision)
            input_clip = clip[:, :-3]
            gt = clip[:, -3:]

            # original
            pred = model(input_clip)
            loss1 = F.mse_loss(pred, gt, reduction='mean')

            # rotated (robust to upside-down corruption)
            input_rot = rotate_180(input_clip)
            gt_rot = rotate_180(gt)
            pred_rot = model(input_rot)
            loss2 = F.mse_loss(pred_rot, gt_rot, reduction='mean')

            loss = min(loss1.item(), loss2.item())
            scores.append(loss)

        scores = np.array(scores, dtype=np.float32)

        # pad first frames (unpredictable frames)
        pad = np.full((4,), scores[0], dtype=np.float32)
        scores = np.concatenate([pad, scores])

        np.save(f'results/{vid:02d}.npy', scores)

    print("Inference completed. Results saved in /results")


if __name__ == '__main__':
    main()
