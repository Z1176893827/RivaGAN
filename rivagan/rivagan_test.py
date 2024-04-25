import gc
import json
import os
import random
import time
from itertools import chain

import cv2
import numpy as np
import pandas as pd
import torch



from torch import optim
from torch.nn import functional
from tqdm import tqdm

from rivagan.adversary import Adversary, Critic
from rivagan.attention_3D4_CASA_chuanxing import AttentiveDecoder, AttentiveEncoder
from rivagan.dataloader_test import load_test
from rivagan.dense import DenseDecoder, DenseEncoder
from rivagan.noise import Compression, Crop, Scale, Rot
from rivagan.utils import mjpeg, psnr, ssim


def get_acc(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()


def quantize(frames):
    # [-1.0, 1.0] -> {0, 255} -> [-1.0, 1.0]
    return ((frames + 1.0) * 127.5).int().float() / 127.5 - 1.0


class RivaGAN(object):

    def __init__(self, model="attention", data_dim=32):
        self.model = model
        self.data_dim = data_dim
        self.adversary = Adversary().cuda()
        self.critic = Critic().cuda()
        if model == "attention":
            self.encoder = AttentiveEncoder(data_dim=data_dim).cuda()
            self.decoder = AttentiveDecoder(self.encoder).cuda()
        elif model == "dense":
            self.encoder = DenseEncoder(data_dim=data_dim).cuda()
            self.decoder = DenseDecoder(data_dim=data_dim).cuda()
        else:
            raise ValueError("Unknown model: %s" % model)

    def fit(self, dataset, log_dir=False,
            seq_len=1, epochs=3, ):
        print("fit函数开始运行")
        if not log_dir:
            log_dir = "experiments/%s-%s" % (self.model, str(int(time.time())))
        os.makedirs(log_dir, exist_ok=False)

        # Set up the noise layers
        crop = Crop()
        scale = Scale()
        compress = Compression()
        rot = Rot()
        i = False

        # Set up the data and optimizers
        test = load_test(seq_len, dataset)

        # Set up the log directory


        # Optimize the model
        history = []
        for epoch in range(1, epochs + 1):
            metrics = {
                "train.loss": [],
                "train.raw_acc": [],
                "train.mjpeg_acc": [],
                "train.adv_loss": [],
                "val.ssim": [],
                "val.psnr": [],
                "val.crop_acc": [],
                "val.scale_acc": [],
                "val.mjpeg_acc": [],
                # uanzhuan
                "val.rot_acc": [],
                "val.yuan_acc": [],
            }
            # Validate
            gc.collect()
            self.encoder.eval()
            self.decoder.eval()
            iterator = tqdm(test, ncols=0)
            with torch.no_grad():
                for frames in iterator:
                    frames = frames.cuda()

                    data = torch.zeros((frames.size(0), self.data_dim)).random_(0, 2).cuda()
                    # print(data)
         #            data = torch.tensor([[1., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
         # 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0.
         #                                  ]]).cuda()#
         #            data = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         # 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
         #                                  ]]).cuda()
         #            data = torch.tensor([[0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0.,
         # 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0.
         #                                  ]]).cuda()
                    wm_frames = self.encoder(frames, data)
                    # print(wm_frames.size())

                    # if not i:
                        # save_images(frames, wm_frames)
                        # save_images(frames, crop(wm_frames))
                        # save_images(frames, scale(wm_frames))
                        # save_images(frames, mjpeg(wm_frames))
                        # i = True

                    wm_crop_data = self.decoder(mjpeg(crop(wm_frames)))

                    # xuanzhuan
                    wm_rot_data = self.decoder(mjpeg(rot(wm_frames)))
                    wm_scale_data = self.decoder(mjpeg(scale(wm_frames)))
                    wm_mjpeg_data = self.decoder(mjpeg(wm_frames))
                    wm_data = self.decoder(wm_frames)

                    metrics["val.ssim"].append(
                        ssim(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())
                    metrics["val.psnr"].append(
                        psnr(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())
                    metrics["val.crop_acc"].append(get_acc(data, wm_crop_data))
                    metrics["val.scale_acc"].append(get_acc(data, wm_scale_data))
                    # xuanzhuan
                    metrics["val.rot_acc"].append(get_acc(data, wm_rot_data))
                    metrics["val.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))
                    metrics["val.yuan_acc"].append(get_acc(data, wm_data))

                    iterator.set_description(
                        "%s | SSIM %.3f | PSNR %.3f | Crop %.5f | Scale %.5f | MJPEG %.5f | Rot %.3f| Acc %.8f" % (
                            epoch,
                            np.mean(metrics["val.ssim"]),
                            np.mean(metrics["val.psnr"]),
                            np.mean(metrics["val.crop_acc"]),
                            np.mean(metrics["val.scale_acc"]),
                            np.mean(metrics["val.mjpeg_acc"]),
                            np.mean(metrics["val.rot_acc"]),
                            np.mean(metrics["val.yuan_acc"]),
                        )
                    )

            metrics = {
                k: round(np.mean(v), 5) if len(v) > 0 else "NaN"
                for k, v in metrics.items()
            }
            # metrics["epoch"] = epoch
            history.append(metrics)
            pd.DataFrame(history).to_csv(
                os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
            with open(os.path.join(log_dir, "metrics.json"), "wt") as fout:
                fout.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))

            # torch.save(self, os.path.join(log_dir, "model_3D2.pt"))
            # G_scheduler.step(metrics["train.loss"])

        return history

    def save(self, path_to_model):
        torch.save(self, path_to_model)

    def load(path_to_model):
        return torch.load(path_to_model)

    def encode(self, video_in, data, video_out):
        assert len(data) == self.data_dim
        print("编码器开始运行")
        video_in = cv2.VideoCapture(video_in)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

        data = torch.FloatTensor([data]).cuda()
        video_out = cv2.VideoWriter(
            video_out, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

        for i in tqdm(range(length)):
            ok, frame = video_in.read()
            cv2.imwrite("image.png",frame)
            # print("读取到视频帧")
            frame = torch.FloatTensor([frame])

            frame = frame / 127.5 - 1.0  # (L, H, W, 3)
            frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()  # (1, 3, L, H, W)

            wm_frame = self.encoder(frame, data)  # (1, 3, L, H, W)

            wm_frame = torch.clamp(wm_frame, min=-1.0, max=1.0)
            wm_frame = (
                (wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
            ).detach().cpu().numpy().astype("uint8")
            # save_images(wm_frame)


            video_out.write(wm_frame)

        video_out.release()

    def decode(self, video_in):
        video_in = cv2.VideoCapture(video_in)
        # width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(length)):
            ok, frame = video_in.read()
            frame = torch.FloatTensor([frame]) / 127.5 - 1.0  # (L, H, W, 3)
            frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()  # (1, 3, L, H, W)
            data = self.decoder(frame)[0].detach().cpu().numpy()
            yield data


# def save_images(watermarked_images):
#
#     watermarked_images = torch.clamp(watermarked_images, min=-1.0, max=1.0)
#
#     watermarked_images = (
#         (watermarked_images[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
#     ).detach().cpu().numpy().astype("uint8")
#     cv2.imwrite("watermarked.png", watermarked_images)


def save_images(original_images, watermarked_images):


    original_images = torch.clamp(original_images, min=-1.0, max=1.0)
    watermarked_images = torch.clamp(watermarked_images, min=-1.0, max=1.0)
    original_images = (
        (original_images[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
    ).detach().cpu().numpy().astype("uint8")
    # cv2.imwrite("base_"+"original_images.png", original_images)
    cv2.imwrite("CASA_"+"original1_mjpeg.png", original_images)
    # cv2.imwrite("multiScale_"+"original_images1_mjpeg.png", original_images)

    watermarked_images = (
        (watermarked_images[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
    ).detach().cpu().numpy().astype("uint8")
    # cv2.imwrite("base_"+"watermarked_images.png", watermarked_images)
    cv2.imwrite("CASA_"+"watermarked1_mjpeg.png", watermarked_images)
    # cv2.imwrite("multiScale_"+"watermarked_images1_mjpeg.png", watermarked_images)




if __name__ == "__main__":

#  模型文件
    model = RivaGAN.load("model_98.pt")

#  视频路径
    model.fit("/home/izuo/pystation/RivaGAN/data/hollywood2")

# 调用UI界面

