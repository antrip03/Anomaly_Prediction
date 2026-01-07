#!/usr/bin/env python
# -*- coding:utf-8 -*-

from glob import glob
import os

# Create necessary folders (Kaggle-safe)
for folder in ['tensorboard_log', 'weights', 'results']:
    if not os.path.exists(folder):
        os.mkdir(folder)

# ============================
# Shared configuration
# ============================
share_config = {
    'mode': 'test',
    'dataset': 'avenue',
    'img_size': (256, 256),

    # 🔥 Kaggle dataset root (IMPORTANT)
    'data_root': '/kaggle/input/pixel-play-26/Avenue_Corrupted-20251221T112159Z-3-001/Avenue_Corrupted/Dataset/'
}  # MUST end with '/'


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None, mode=None):
    """
    Only TEST mode is used in this competition.
    Training-related options are kept for compatibility but unused.
    """

    # ----------------------------
    # Basic setup
    # ----------------------------
    share_config['mode'] = mode
    assert args.dataset in ('ped2', 'avenue', 'shanghaitech'), 'Dataset error.'
    share_config['dataset'] = args.dataset

    # ----------------------------
    # TEST MODE (USED)
    # ----------------------------
    if mode == 'test':
        share_config['test_data'] = (
            share_config['data_root'] + 'testing_videos/'
        )

        # 🔥 Pretrained Avenue model
        share_config['trained_model'] = args.trained_model

        # Visualization flags (safe to keep False)
        share_config['show_curve'] = False
        share_config['show_heatmap'] = False

    # ----------------------------
    # TRAIN MODE (NOT USED, KEPT SAFE)
    # ----------------------------
    elif mode == 'train':
        share_config['batch_size'] = args.batch_size
        share_config['train_data'] = (
            share_config['data_root'] + 'training_videos/'
        )
        share_config['test_data'] = (
            share_config['data_root'] + 'testing_videos/'
        )

        share_config['g_lr'] = 0.0002
        share_config['d_lr'] = 0.00002
        share_config['resume'] = (
            glob(f'weights/{args.resume}*')[0] if args.resume else None
        )
        share_config['iters'] = args.iters
        share_config['show_flow'] = args.show_flow
        share_config['save_interval'] = args.save_interval
        share_config['val_interval'] = args.val_interval
        share_config['flownet'] = args.flownet

    return dict2class(share_config)
