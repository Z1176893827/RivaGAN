from random import randint, random

import cv2
import random as rand
import torch
import torch_dct as dct

from torch import nn

import numpy as np


class Crop(nn.Module):
    """
    Randomly crops the two spatial dimensions independently to a new size
    that is between `min_pct` and `max_pct` of the old size.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H', W')
    """

    def __init__(self, min_pct=0.8, max_pct=1.0):
        super(Crop, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def _pct(self):
        return self.min_pct + random() * (self.max_pct - self.min_pct)

    def forward(self, frames):
        _, _, _, height, width = frames.size()
        dx = int(self._pct() * width)
        dy = int(self._pct() * height)
        dx, dy = (dx // 4) * 4, (dy // 4) * 4
        x = randint(0, width - dx - 1)
        y = randint(0, height - dy - 1)
        return frames[:, :, :, y:y + dy, x:x + dx]

class GaussianNoise(nn.Module):
    def __init__(self,min_std_dev=0.02, max_std_dev=0.06):
        super(GaussianNoise, self).__init__()
        self.min_std_dev = min_std_dev
        self.max_std_dev = max_std_dev

    def _std_dev(self):

        return self.min_std_dev + random() * (self.max_std_dev - self.min_std_dev)


    def forward(self, frames):
        # 生成与输入帧相同大小的高斯噪声
        noise = torch.randn(frames.size()).cuda() * self._std_dev()

        # 将噪声添加到输入帧中
        noisy_frames = frames + noise
        return noisy_frames

class Scale(nn.Module):
    """
    Randomly scales the two spatial dimensions independently to a new size
    that is between `min_pct` and `max_pct` of the old size.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_pct=0.8, max_pct=1.0):
        super(Scale, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def _pct(self):
        return self.min_pct + random() * (self.max_pct - self.min_pct)

    def forward(self, frames):
        # print("scale_frames", frames.size())  # ([24, 3, 1, 160, 160])
        percent = self._pct()
        _, _, depth, height, width = frames.size()
        height, width = int(percent * height), int(percent * width)
        height, width = (height // 4) * 4, (width // 4) * 4
        return nn.AdaptiveAvgPool3d((depth, height, width))(frames)


class Rot(nn.Module):
    """
    Randomly scales the two spatial dimensions independently to a new size
    that is between `min_pct` and `max_pct` of the old size.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_rot=0, max_rot=10):
        super(Rot, self).__init__()
        self.min_rot = min_rot
        self.max_rot = max_rot

    def _pct(self):
        return self.min_rot + random() * (self.max_rot - self.min_rot)

    def forward(self, frames):
        # print("rot_in_frames",frames.size())
        # N, D, L, H, W = frames.size()
        # frame = torch.clamp(frames, min=-1.0, max=1.0)
        # frame = (  # N C L H W
        #     (frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5  # 将像素值-1.0~1.0 转到0-255
        # ).detach().cpu().numpy().astype("uint8")
        # for i in range(N):
        #     rows, cols = frame.shape[:2]
        #     angle = rand.randint(10, 10)  # 旋转方向取（-180，180）中的随机整数值，负为逆时针，正为顺势针
        #     scale = 1.0  # 0.8 将图像缩放为  80 %
        #     M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
        #
        #     frame = cv2.warpAffine(frame, M, (cols, rows))
        #
        #     frame = torch.FloatTensor([frame]) / 127.5 - 1.0  # (L, H, W, 3)
        #     frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()  # (1, 3, L, H, W)
        #     out_frame = torch.cat(frame,dim=0)
        # print("rot_out_frames" , frames.size()) # torch.Size([24, 3, 1, 156, 148])

        # print("rot_in_frames",frames.size())
        batch_size, num_channels, seq_len, height, width = frames.shape
        frames = torch.clamp(frames, min=-1.0, max=1.0)
        output_frames = []
        for i in range(batch_size):
            frame = (frames[i, :, 0, :, :].permute(1, 2,
                                                   0).detach().cpu().numpy() + 1.0) * 127.5  # 将像素值-1.0~1.0 转到0-255
            rows, cols = frame.shape[:2]
            angle = rand.randint(self.min_rot, self.max_rot)
            scale = 1.0  # 0.8 将图像缩放为  80 %
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
            frame = cv2.warpAffine(frame, M, (cols, rows))
            frame = (torch.FloatTensor(frame).permute(2, 0, 1) / 127.5 - 1.0).unsqueeze(
                0)  # (1, C, H, W)
            output_frames.append(frame)
        output_frames = torch.cat(output_frames, dim=0)  # (batch_size, C, H, W)
        output_frames = output_frames.unsqueeze(2).repeat(1, 1, seq_len, 1,1).cuda()  # (batch_size, C, seq_len, H, W)
        # print("out",output_frames.size()) #  out torch.Size([1, 3, 1, 360, 480])

        return output_frames


class Compression(nn.Module):
    """
    This uses the DCT to produce a differentiable approximation of JPEG compression.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, yuv=False, min_pct=0.0, max_pct=0.5):
        super(Compression, self).__init__()
        self.yuv = yuv
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, y):
        N, _, L, H, W = y.size()

        L = int(y.size(2) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        H = int(y.size(3) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        W = int(y.size(4) * (random() * (self.max_pct - self.min_pct) + self.min_pct))

        if self.yuv:
            y = torch.stack([
                (0.299 * y[:, 2, :, :, :] +
                 0.587 * y[:, 1, :, :, :] +
                 0.114 * y[:, 0, :, :, :]),
                (- 0.168736 * y[:, 2, :, :, :] -
                 0.331264 * y[:, 1, :, :, :] +
                 0.500 * y[:, 0, :, :, :]),
                (0.500 * y[:, 2, :, :, :] -
                 0.418688 * y[:, 1, :, :, :] -
                 0.081312 * y[:, 0, :, :, :]),
            ], dim=1)

        y = dct.dct_3d(y)

        if L > 0:
            y[:, :, -L:, :, :] = 0.0

        if H > 0:
            y[:, :, :, -H:, :] = 0.0

        if W > 0:
            y[:, :, :, :, -W:] = 0.0

        y = dct.idct_3d(y)

        if self.yuv:
            y = torch.stack([
                (1.0 * y[:, 0, :, :, :] +
                 1.772 * y[:, 1, :, :, :] +
                 0.000 * y[:, 2, :, :, :]),
                (1.0 * y[:, 0, :, :, :] -
                 0.344136 * y[:, 1, :, :, :] -
                 0.714136 * y[:, 2, :, :, :]),
                (1.0 * y[:, 0, :, :, :] +
                 0.000 * y[:, 1, :, :, :] +
                 1.402 * y[:, 2, :, :, :]),
            ], dim=1)

        return y
