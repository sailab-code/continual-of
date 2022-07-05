import glob
import os
from gzip import GzipFile

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class PairsOfFramesDataset(Dataset):
    # example class exploiting the prerendered streams
    # additional preprocessing options:
    # force_gray (grayscale conversion)
    # invert_gt_flow_u, invert_gt_flow_v in case you need to adjust for reference axes
    def __init__(self, root_dir, device, force_gray=True, invert_gt_flow_u=False, invert_gt_flow_v=False):
        self.root_dir = root_dir
        self.files = sorted(glob.glob(root_dir + os.sep + "frames" + os.sep + "**" + os.sep + "*.png", recursive=True))
        self.motion_files = sorted(glob.glob(root_dir + os.sep + "motion" + os.sep + "**" + os.sep + "*.bin", recursive=True))
        self.motion_available = len(self.motion_files) > 0
        self.force_gray = force_gray
        self.length = len(self.files) - 1  # remove last frame
        self.device = device
        self.invert_gt_flow_v = invert_gt_flow_v
        self.invert_gt_flow_u = invert_gt_flow_u

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        old_frame = cv2.imread(self.files[idx])
        frame = cv2.imread(self.files[idx + 1])

        if self.force_gray and frame.shape[2] > 1:
            frame = np.reshape(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (frame.shape[0], frame.shape[1], 1))

        if self.motion_available:
            with GzipFile(self.motion_files[idx + 1]) as f:
                motion = np.load(f)
                motion = torch.from_numpy(motion.transpose(2, 0, 1)).float()
                if self.invert_gt_flow_v: motion[1, :, :] *= -1
                if self.invert_gt_flow_u: motion[0, :, :] *= -1
        else:
            motion = torch.empty(1)

        if self.force_gray and old_frame.shape[2] > 1:
            old_frame = np.reshape(cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY),
                                   (old_frame.shape[0], old_frame.shape[1], 1))

        frame = torch.from_numpy(frame.transpose(2, 0, 1)).float().div_(255.0)
        old_frame = torch.from_numpy(old_frame.transpose(2, 0, 1)).float().div_(255.0)
        return (old_frame, frame, motion)