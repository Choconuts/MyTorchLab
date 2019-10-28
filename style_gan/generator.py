import torch
import numpy as np
import math
from torch import nn, Tensor
from torch.nn import functional as nnf

from style_gan.config import *
from style_gan.gblock import *
from style_gan.map_net import *


class Generator(nn.Module):

    def __init__(self, stage=0):
        super().__init__()

        def set_block(i):
            self.__setattr__('block_%d' % i, GBlock(i))
            self.__setattr__('affine_%d_1' % i, nn.Linear(opt.latent_size, opt.fm_chan * 2))
            self.__setattr__('affine_%d_2' % i, nn.Linear(opt.latent_size, opt.fm_chan * 2))

        self.get_block = lambda i: getattr(self, 'block_%d' % i)
        self.get_affine = lambda i, k: getattr(self, 'affine_%d_%d' % (i, k))
        for i in range(9):
            set_block(i)

        self.final_conv = nn.Conv2d(opt.fm_chan, opt.im_chan, 1)

        self.map_net = MapNet(opt.fc_units, 8)

        self.start = nn.Parameter(torch.zeros(opt.fm_chan, 4, 4), True)

        self.stage = stage
        self.alpha = 0

        if opt.spec_norm:
            self.final_conv = nn.utils.spectral_norm(self.final_conv)
            # self.map_net = nn.utils.spectral_norm(self.map_net)

    def forward(self, z: Tensor):
        w = self.map_net(z)
        const_start = self.start + torch.zeros(z.shape[0], *self.start.shape).to(self.start.device)

        nxt = const_start
        lst = nxt
        for i in range(self.stage + 1):
            tmp = nxt
            nxt = self.get_block(i)(nxt, self.get_affine(i, 1)(w), self.get_affine(i, 2)(w))
            lst = tmp
        if self.stage >= 0:
            if self.stage >= 1:
                nxt0 = nnf.interpolate(lst, scale_factor=2)
            else:
                nxt0 = lst
            out = nxt0 * (1 - self.alpha) + nxt * self.alpha
        else:
            out = nxt
        out = self.final_conv(out)
        return torch.tanh(out)

    def forwardbk(self, z: Tensor):
        nxt = z
        lst = nxt
        for i in range(self.stage + 1):
            tmp = nxt
            nxt = self.get_block(i)(nxt, None, None)
            lst = tmp
        if self.stage >= 1:
            nxt0 = nnf.interpolate(lst, scale_factor=2)
            out = nxt0 * (1 - self.alpha) + nxt * self.alpha
        else:
            out = nxt
        out = self.final_conv(out)
        return torch.tanh(out)


class Generator0(nn.Module):

    def __init__(self, stage):
        super(Generator0, self).__init__()

        self.code2image = nn.Sequential(
            nn.Linear(opt.latent_size, 64),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.1),
        )
        a = 8
        self.linear_8_to_16 = nn.Sequential(
            nn.Linear(a*a, a*a*4),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.1),
        )
        a = 16
        self.linear_16_to_32 = nn.Sequential(
                nn.Linear(a*a, a*a*4),
                nn.LeakyReLU(0.2),
                # nn.Dropout(0.1),
            )
        self.conv_2d = nn.Conv2d(1, 1, 1)
        self.stage = stage
        self.alpha = 0

    def conv(self, img):
        b, s = img.shape
        ss = int(np.sqrt(s))
        return self.conv_2d(img.view(b, 1, ss, ss)).view(b, -1)

    def forward(self, z):
        batch_size = z.size(0)
        z = z.view(z.size(0), -1)
        img_8 = self.code2image(z).view(batch_size, 64)
        img_8 = self.conv(img_8)
        if self.stage == 1:
            return torch.tanh(img_8.view(batch_size, 1, 8, 8))
        img_16 = self.linear_8_to_16(img_8)
        img_16 = self.conv(img_16)
        if self.stage == 2:
            img_8_8_ = img_8.view(batch_size, 1, 8, 8)
            img_16_16 = nn.functional.interpolate(img_8_8_, scale_factor=2)
            combine = img_16_16.view(batch_size, -1) * (1 - self.alpha) + img_16 * self.alpha
            return torch.tanh(combine.view(batch_size, 1, 16, 16))
        img_32 = self.linear_16_to_32(img_16)
        img_32 = self.conv(img_32)
        if self.stage == 3:
            img_16_16_ = img_16.view(batch_size, 1, 16, 16)
            img_32_32 = nn.functional.interpolate(img_16_16_, scale_factor=2)
            combine = img_32_32.view(batch_size, -1) * (1 - self.alpha) + img_32 * self.alpha
            return torch.tanh(combine.view(batch_size, 1, 32, 32))


class Generator(nn.Module):

    def __init__(self, stage=0):
        super().__init__()

        def set_block(i):
            self.__setattr__('block_%d' % i, GBlock(i))

        self.get_block = lambda i: getattr(self, 'block_%d' % i)
        for i in range(9):
            set_block(i)

        self.final_conv = nn.Conv2d(opt.fm_chan, opt.im_chan, 1)

        self.map_net = GMapping(opt.fm_chan, opt.latent_size, resolution=opt.im_res, normalize_latents=True)

        self.start = nn.Parameter(torch.zeros(1, opt.fm_chan, 4, 4), True)

        self.stage = stage
        self.alpha = 0

        if opt.spec_norm:
            self.final_conv = nn.utils.spectral_norm(self.final_conv)
            # self.map_net = nn.utils.spectral_norm(self.map_net)

    def forward(self, z: Tensor):
        w = self.map_net(z)
        const_start = self.start.expand(z.size(0), opt.fm_chan, 4, 4)

        nxt = const_start
        lst = nxt
        for i in range(self.stage + 1):
            tmp = nxt
            nxt = self.get_block(i)(nxt, w)
            lst = tmp
        if self.stage >= 0:
            if self.stage >= 1:
                nxt0 = nnf.interpolate(lst, scale_factor=2)
            else:
                nxt0 = lst
            out = nxt0 * (1 - self.alpha) + nxt * self.alpha
        else:
            out = nxt
        out = self.final_conv(out)
        return torch.tanh(out)

    def forwardbk(self, z: Tensor):
        nxt = z
        lst = nxt
        for i in range(self.stage + 1):
            tmp = nxt
            nxt = self.get_block(i)(nxt, None)
            lst = tmp
        if self.stage >= 1:
            nxt0 = nnf.interpolate(lst, scale_factor=2)
            out = nxt0 * (1 - self.alpha) + nxt * self.alpha
        else:
            out = nxt
        out = self.final_conv(out)
        return torch.tanh(out)