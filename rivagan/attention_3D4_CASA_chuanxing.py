import torch
from torch import nn
from torch.nn import functional

from CBAM import CBAMBlock


def multiplicative(x, data):
    """
    This function takes a 5d tensor (with the same shape and dimension order
    as the input to Conv3d) and a 2d data tensor. For each element in the
    batch, the data vector is combined with the first D dimensions of the 5d
    tensor through an elementwise product.

    Input: (N, C_{in}, L, H, W), (N, D)
    Output: (N, C_{in}, L, H, W)
    """
    N, D = data.size()
    N, C, L, H, W = x.size()
    assert D <= C, "data dims must be less than channel dims"
    x = torch.cat([
        x[:, :D, :, :, :] * data.view(N, D, 1, 1, 1).expand(N, D, L, H, W),
        x[:, D:, :, :, :]
    ], dim=1)
    return x


class AttentiveEncoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D, )
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim, tie_rgb=False, linf_max=0.016,
                 kernel_size=(1, 11, 11), padding=(0, 5, 5),

                 kernel_size3=(1, 9, 9), padding3=(0, 4, 4),
                 kernel_size2=(1, 7, 7), padding2=(0, 3, 3),
                 kernel_size1=(1, 5, 5), padding1=(0, 2, 2),
                 ):
        super(AttentiveEncoder, self).__init__()

        self.linf_max = linf_max
        self.data_dim = data_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.kernel_size1 = kernel_size1
        self.padding1 = padding1
        self.kernel_size2 = kernel_size2
        self.padding2 = padding2
        self.kernel_size3 = kernel_size3
        self.padding3 = padding3
        self.cbam = CBAMBlock(data_dim)

        self._attention = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=kernel_size1, padding=padding1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            CBAMBlock(32),
            nn.Conv3d(32, 64, kernel_size=kernel_size1, padding=padding1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            CBAMBlock(64),
            nn.Conv3d(64, 3, kernel_size=kernel_size1, padding=padding1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(3),

        )
        self._attention2 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            CBAMBlock(32),
            nn.Conv3d(32, 64, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            CBAMBlock(64),
            nn.Conv3d(64, 3, kernel_size=kernel_size2, padding=padding2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(3),
        )
        self._attention3 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=kernel_size3, padding=padding3),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            CBAMBlock(32),
            nn.Conv3d(32, 64, kernel_size=kernel_size3, padding=padding3),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            CBAMBlock(64),

            nn.Conv3d(64, 32, kernel_size=kernel_size3, padding=padding3),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
        )

        self._conv1x1 = nn.Sequential(
            nn.Conv3d(32, 32, kernel_size=(1, 1, 1), padding=0),
            # nn.ReLU(inplace=True),
            # nn.BatchNorm3d(32),
        )

        self._conv = nn.Sequential(
            nn.Conv3d(10, 32, kernel_size=(1,1,1), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=(1,1,1), padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 1 if tie_rgb else 3, kernel_size=(1,1,1), padding=0),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        data = data * 2.0 - 1.0

        frames1 = self._attention(frames)

        f1 = frames1 * frames
        frames1 = self._attention2(f1)

        f2 = frames1 * frames
        frames1 = self._attention3(f2)

        # f3 = frames1 * frames
        # frames1 = self._attention4(f3)
        # f4 = self._att4(f3)

        x = self.cbam(frames1)
        x = self._conv1x1(x)

        # x = functional.softmax(frames1, dim=1)

        x = torch.sum(multiplicative(x, data), dim=1, keepdim=True)
        x = self._conv(torch.cat([frames, f1, f2, x], dim=1))

        # return frames + 0.014 * x
        return frames + self.linf_max * x


class AttentiveDecoder(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, D)
    """

    def __init__(self, encoder):
        super(AttentiveDecoder, self).__init__()
        self.data_dim = encoder.data_dim
        self._attention = encoder._attention

        self.cbam = encoder.cbam
        self._conv1x1 = encoder._conv1x1

        self._attention = encoder._attention
        self._attention2 = encoder._attention2
        self._attention3 = encoder._attention3
        # self._attention4 = encoder._attention4

        self._conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=encoder.kernel_size1, padding=encoder.padding1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            CBAMBlock(32),
            nn.Conv3d(32, 64, kernel_size=encoder.kernel_size1, padding=encoder.padding1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            CBAMBlock(64),

            nn.Conv3d(64, 3, kernel_size=encoder.kernel_size1,
                      padding=encoder.padding1, stride=1),

        )
        self._conv2 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=encoder.kernel_size2, padding=encoder.padding2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            CBAMBlock(32),
            nn.Conv3d(32, 64, kernel_size=encoder.kernel_size2, padding=encoder.padding2, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            CBAMBlock(64),
            nn.Conv3d(64, 3, kernel_size=encoder.kernel_size2,
                      padding=encoder.padding2, stride=1),

        )
        self._conv3 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=encoder.kernel_size3, padding=encoder.padding3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(32),
            CBAMBlock(32),
            nn.Conv3d(32, 64, kernel_size=encoder.kernel_size3, padding=encoder.padding3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(64),
            CBAMBlock(64),

            nn.Conv3d(64, 32, kernel_size=encoder.kernel_size3,
                      padding=encoder.padding3, stride=1),

        )
        # self._conv4 = nn.Sequential(
        #     nn.Conv3d(3, 32, kernel_size=encoder.kernel_size, padding=encoder.padding, stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.BatchNorm3d(32),
        #     CBAMBlock(32),
        #
        #     nn.Conv3d(32, self.data_dim, kernel_size=encoder.kernel_size,
        #               padding=encoder.padding, stride=1),
        # )

    def forward(self, frames):
        N, D, L, H, W = frames.size()
        frames1 = self._attention(frames) * frames

        frames1 = self._attention2(frames1) * frames

        frames1 = self._attention3(frames1)

        # frames1 = self._attention4(frames1)

        attention = self.cbam(frames1)
        attention = self._conv1x1(attention)

        # attention = functional.softmax(frames1, dim=1)
        f = self._conv(frames)
        # frames = frames + f
        f = self._conv2(frames + f)
        # frames = frames + f
        x = self._conv3(frames + f) * attention
        # frames = frames + f
        # x = self._conv4(frames + f) * attention
        # x = f
        # print("x",x.size())  # x torch.Size([24, 64, 1, 160, 160])
        return torch.mean(x.view(N, self.data_dim, -1), dim=2)
