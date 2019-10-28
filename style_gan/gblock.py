import torch
import numpy as np
import math
from torch import nn, Tensor
from torch.nn import functional as nnf

from style_gan.config import *
from style_gan.ops import *


class GLayer(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,

                 epsilon=1e-8,
                 no_conv=False,
                 no_noise=False,
                 use_norm='A'                # N for None, I for Instance, P for Pixel and A for AdaIN
                 ):
        super().__init__()

        self.no_conv = no_conv
        self.no_noise = no_noise
        self.use_pixel_norm = use_norm is 'P'
        self.use_instance_norm = use_norm is 'A' or use_norm is 'I'
        self.use_style = use_norm is 'A'

        """ Conv2d """
        if not self.no_conv:
            self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=(1, 1))
            if opt.spec_norm:
                self.conv = nn.utils.spectral_norm(self.conv)

        """ Pixel Norm """
        if self.use_pixel_norm:
            # self.pixel_norm = nn.LocalResponseNorm(1)
            self.pixel_norm = PixelNorm(epsilon)

        """ Instance Norm """
        if self.use_instance_norm:
            # self.instance_norm = nn.InstanceNorm2d(out_channel, epsilon, affine=False)
            self.instance_norm = InstanceNorm(epsilon)

        """ Add Noise """
        if not self.no_noise:
            self.noise = ApplyScaledNoise(out_channel)

        """ Apply Style """
        if self.use_style:
            self.style = ApplyStyleAffine(opt.latent_size, out_channel, opt.use_ws)

    def forward(self, img: Tensor, w_latent: Tensor = None, noise: Tensor = None):
        # convolution
        x = img if self.no_conv else self.conv(img)

        # add noise
        x = x if self.no_noise else self.noise(x, noise)
        # activation
        x = nnf.leaky_relu(x, negative_slope=opt.lrelu)
        # norm
        x = x if self.use_pixel_norm else self.instance_norm(x)
        x = x if self.use_instance_norm else self.pixel_norm(x)

        # style
        x = x if not self.use_style else self.style(x, w_latent)

        # if affine is not None:
        #     x = ((affine[:, 0] + 1.) * x.permute([2, 3, 0, 1]) + affine[:, 1]).permute([2, 3, 0, 1])
        return x


class GBlock(nn.Module):

    def __init__(self,
                 stage
                 ):
        super().__init__()
        self.stage = stage
        # self.nf = lambda st: opt.fm_res if stage <= opt.st_thr \
        #     else int(opt.fm_res // 2 ** (stage - opt.st_thr))
        self.nf = lambda st: opt.fm_chan

        self.fm = self.nf(stage)
        self.fm0 = self.nf(stage - 1)
        self.linear = nn.Sequential(
            nn.Linear(opt.latent_size, opt.res_0 * opt.res_0 * self.fm),
            nn.Dropout(opt.dropout),
            nn.LeakyReLU(opt.lrelu, inplace=True),
        )
        no_noise = opt.noise in ['no', 'not', 'none']
        self.layer_1 = GLayer(
            in_channel=self.fm,
            out_channel=self.fm,
            no_conv=(stage is 0),
            use_norm=opt.norm,
            no_noise=no_noise,
        )
        self.layer_2 = GLayer(
            in_channel=self.fm,
            out_channel=self.fm,
            use_norm=opt.norm,
            no_noise=no_noise,
        )
        self.layer_3 = GLayer(
            in_channel=self.fm,
            out_channel=self.fm,
            use_norm=opt.norm,
            no_noise=no_noise,
        )

        if opt.noise == 'fix':
            self.noise = torch.randn(2, 1, 1, opt.res_0 * 2 ** self.stage)

        self.blur = Blur2d()

        if stage < 5:
            # upsample method 1
            self.up_sample = Upscale2d(2)
        else:
            # upsample method 2
            self.up_sample = nn.ConvTranspose2d(self.nf(stage - 1), self.nf(stage), 4, stride=2, padding=1)

    # def forward(self, img: Tensor, affine_1: Tensor = None, affine_2: Tensor = None, noise=None):
    #     batch_size = img.size(0)
    #
    #     if self.stage == 0:
    #         if affine_1 is None:
    #             img = self.linear(img.view(batch_size, -1)).view(batch_size, self.fm, 4, 4)
    #     else:
    #         img = nnf.interpolate(img, scale_factor=2)
    #
    #     img = self.blur(img)
    #
    #     img_1 = self.layer_1(img, None if affine_1 is None else affine_1.view(batch_size, 2, -1), noise)
    #     # save_sum(affine_1, 0)
    #     img_2 = self.layer_2(img_1, None if affine_2 is None else affine_2.view(batch_size, 2, -1), noise)
    #     # save_sum(affine_2, 1)
    #     return img_2

    def forward(self, img: Tensor, w_latent: Tensor = None):
        batch_size = img.size(0)
        if opt.noise == 'fix':
            noises = self.noise
        else:
            noises = [None, None]

        if self.stage == 0:
            if not opt.const_start:
                img = self.linear(img.view(batch_size, -1)).view(batch_size, self.fm, 4, 4)
        else:
            if opt.bilinear:
                img = nnf.interpolate(img, scale_factor=2)
            else:
                img = self.up_sample(img)

        if opt.blur:
            img = self.blur(img)
        img = self.layer_1(img, w_latent, noises[0])
        # save_sum(img, 0)
        img = self.layer_2(img, w_latent, noises[1])
        # save_sum(img, 1)
        return img

