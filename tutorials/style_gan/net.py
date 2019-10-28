import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn, Tensor
from torch.nn import functional as nnf

from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from tutorials.style_gan.config import *
from style_gan.ops import *


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels), requires_grad=False)

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class GLayer(nn.Module):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size=3,
                 epsilon=1e-8,
                 no_conv=False,
                 ):
        super(GLayer, self).__init__()
        self.no_conv = no_conv
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=(1, 1))
        # self.pixel_norm = nn.LocalResponseNorm(1)
        self.pixel_norm = PixelNorm()

        # self.instance_norm = nn.InstanceNorm2d(out_channel, epsilon, affine=False)
        self.instance_norm = InstanceNorm()
        self.noise = ApplyScaledNoise(out_channel)

    def forward(self, img: Tensor, affine: Tensor = None):
        c = img if self.no_conv else self.conv(img)
        cn = self.noise(c)
        cn = nnf.leaky_relu(cn, negative_slope=opt.lrelu)
        cnn = self.instance_norm(cn)
        # (80, 512, 32, 32)
        # (80, 2, 512)
        if affine is not None:
            cnn = ((affine[:, 0] + 1.) * cnn.permute([2, 3, 0, 1]) + affine[:, 1]).permute([2, 3, 0, 1])
        return cnn


class GBlock(nn.Module):

    def __init__(self,
                 stage
                 ):
        super(GBlock, self).__init__()
        self.stage = stage
        # self.nf = lambda st: opt.fm_res if stage <= opt.st_thr \
        #     else int(opt.fm_res // 2 ** (stage - opt.st_thr))
        self.nf = lambda st: opt.fm_chan
        self.fm = self.nf(stage)
        self.fm0 = self.nf(stage - 1)
        self.linear = nn.Sequential(
            nn.Linear(opt.latent_size, 4 * 4 * self.fm),
            nn.Dropout(opt.dropout),
            nn.LeakyReLU(opt.lrelu, inplace=True),
        )
        self.norm = GLayer(
            in_channel=self.fm,
            out_channel=self.fm,
            no_conv=True,
        )
        self.layer_1 = GLayer(
            in_channel=self.fm,
            out_channel=self.fm,
        )
        self.layer_2 = GLayer(
            in_channel=self.fm,
            out_channel=self.fm,
        )

    def forward(self, img: Tensor, affine_1=None, affine_2=None):
        if affine_1 is not None:
            batch_size = affine_1.size(0)
        else:
            batch_size = img.shape[0]
        if self.stage == 0:
            if affine_1 is None:
                img = img.view(batch_size, -1)
                img_1 = self.linear(img)
                img_1 = img_1.view(batch_size, self.fm, 4, 4)
            else:
                img_1 = img + torch.zeros(batch_size, *img.shape).to(img.device)
            img_1 = self.norm(img_1)
        else:
            img_1 = nnf.interpolate(img, scale_factor=2)
            img_1 = self.layer_1(img_1, affine_1)
        img_2 = self.layer_2(img_1, affine_2)
        return img_2


class MapNet(nn.Module):

    def __init__(self, units=512, layers=8):
        super(MapNet, self).__init__()

        self.mapping = nn.Sequential()

        def act():
            self.mapping.add_module('Dropout', nn.Dropout(opt.dropout))
            self.mapping.add_module('LeakyReLU', nn.LeakyReLU(opt.lrelu))

        self.mapping.add_module('FC_0', nn.Linear(opt.latent_size, units))
        act()
        for i in range(1, layers - 1):
            self.mapping.add_module('FC_%d' % i, nn.Linear(units, units))
            act()
        self.mapping.add_module('FC_%d' % (layers - 1), nn.Linear(units, opt.latent_size))

        self.simple_mapping = nn.Sequential(
            nn.Linear(opt.latent_size, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),

            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),

            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),

            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Linear(units, opt.latent_size),
            nn.LeakyReLU(opt.lrelu),
        )

    def forward(self, z: Tensor):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)
        w = self.simple_mapping(z)
        return w


class Generator(nn.Module):

    def __init__(self, stage=0):
        super(Generator, self).__init__()

        def set_block(i):
            self.__setattr__('block_' + str(i), GBlock(i))

        self.get_block = lambda i: getattr(self, 'block_' + str(i))
        for i in range(9):
            set_block(i)

        self.final_conv = nn.Sequential(
            nn.Conv2d(opt.fm_chan, opt.im_chan, 1),
            # nn.InstanceNorm2d(opt.im_chan),
            # nn.Tanh(),
        )

        self.map_net = MapNet(opt.fc_units, 8)
        self.to_affine = nn.Sequential(
            nn.Linear(opt.latent_size, opt.fm_chan * 2 * 18),
        )
        self.start = nn.Parameter(torch.zeros(opt.fm_chan, 4, 4), True)

        self.stage = stage
        self.alpha = 0

    def forward0(self, z: Tensor):
        affs = self.to_affine(self.map_net(z)).view(z.shape[0], 18, 2, opt.fm_chan)
        nxt = self.start
        lst = nxt
        for i in range(self.stage + 1):
            tmp = nxt
            nxt = self.get_block(i)(nxt, None, affs[:, i*2+1])
            lst = tmp
        if self.stage >= 1:
            nxt0 = nnf.interpolate(lst, scale_factor=2)
            out = nxt0 * (1 - self.alpha) + nxt * self.alpha
        else:
            out = nxt
        out = self.final_conv(out)
        return torch.tanh(out)

    def forwardfn(self, z: Tensor):
        affs = self.to_affine(self.map_net(z)).view(z.shape[0], 18, 2, opt.fm_chan)
        nxt = z
        lst = nxt
        for i in range(self.stage + 1):
            tmp = nxt
            nxt = self.get_block(i)(nxt, None, affs[:, i*2+1])
            lst = tmp
        if self.stage >= 1:
            nxt0 = nnf.interpolate(lst, scale_factor=2)
            out = nxt0 * (1 - self.alpha) + nxt * self.alpha
        else:
            out = nxt
        out = self.final_conv(out)
        return torch.tanh(out)

    def forward(self, z: Tensor):
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


class Generator_DC(nn.Module):

    def __init__(self, stage):
        super(Generator_DC, self).__init__()

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
