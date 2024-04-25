import numpy as np
import torch
from torch import nn
from torch.nn import init


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=8): # 12bit时reduction用的8
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.se = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        # print(max_result.size()) torch.Size([50, 512, 1, 1])
        avg_result = self.avgpool(x)
        # print(avg_result.size())torch.Size([50, 512, 1, 1])

        max_out = self.se(max_result)

        # print(max_out.size())torch.Size([50, 512, 1, 1])
        avg_out = self.se(avg_result)
        # print(avg_out.size())torch.Size([50, 512, 1, 1])

        output = self.sigmoid(max_out + avg_out)
        # print(output.size()) # ([50, 512, 1, 1])
        return output


class SpatialAttention(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=(1, 7, 7), padding=(0, 3, 3))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output


class CBAMBlock(nn.Module):

    def __init__(self, channel, ):
        super().__init__()
        self.ca = ChannelAttention(channel=channel,)
        self.sa = SpatialAttention()

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             init.kaiming_normal_(m.weight, mode='fan_out')
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             init.constant_(m.weight, 1)
    #             init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.Linear):
    #             init.normal_(m.weight, std=0.001)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _, _ = x.size()
        residual = x
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + residual


