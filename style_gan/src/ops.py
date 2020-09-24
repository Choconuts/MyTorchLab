import torch
import numpy as np
import math
from torch import nn, Tensor
from torch.nn import functional as nnf

from style_gan.config import *


class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std, True)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels), True)
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = nnf.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = nnf.linear(x, self.weight * self.w_lrmul)
        out = nnf.leaky_relu(out, 0.2, inplace=True)
        return out


##################################################################################
# Style-GAN operations
##################################################################################


class ApplyScaledNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels), requires_grad=True)

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class ApplyStyleAffine(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super().__init__()
        self.linear = FC(latent_size,
                         channels * 2,
                         gain=1.0,
                         use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)
        # save_sum(style.view(-1, 2 * x.size(1)), 3)
        x = x * (style[:, 0] + 1.) + style[:, 1]
        # save_sum(x, 4)
        return x


##################################################################################
# Normalizations for 4D tensors
##################################################################################


class PixelNorm(nn.Module):
    """
    normalize on channels
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        """

        :param x: (n, c, h, [w,])
        :return:
        """
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class InstanceNorm(nn.Module):
    """
    normalize on feature maps per channel
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x = x - torch.mean(x, (2, 3), True)
        return x * torch.rsqrt(torch.mean(x ** 2, dim=(2, 3), keepdim=True) + self.epsilon)


##################################################################################
# Filtering for 4D tensors
##################################################################################


class Blur2d(nn.Module):
    def __init__(self, f=(1,2,1), normalize=True, flip=False, stride=1):
        """
            depthwise_conv2d:
            https://blog.csdn.net/mao_xiao_feng/article/details/78003476
        """
        super().__init__()
        assert isinstance(f, (list, tuple)) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = torch.tensor(f, dtype=torch.float32)
            f = f[:, None] * f[None, :]
            f = f[None, None]
            if normalize:
                f = f / f.sum()
            if flip:
                # f = f[:, :, ::-1, ::-1]
                f = torch.flip(f, [2, 3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1).to(x.device)
            x = nnf.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.size(2)-1)/2),
                groups=x.size(1)
            )
            return x
        else:
            return x


class Conv2d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size) * init_std, True)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels), True)
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return nnf.conv2d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return nnf.conv2d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)


class Upscale2d(nn.Module):
    def __init__(self, factor=2, gain=1):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super().__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            # x = nnf.interpolate(x, scale_factor=self.factor)
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1).expand(-1, -1, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3])
        return x