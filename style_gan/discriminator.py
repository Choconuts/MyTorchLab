import torch
import numpy as np
import math
from torch import nn, Tensor
from torch.nn import functional as nnf
from style_gan.config import opt

from style_gan.ops import *

class Discriminator(nn.Module):

    def __init__(self, stage):
        super(Discriminator, self).__init__()
        self.image2score = nn.Sequential(
            nn.Linear(16, 128),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            nn.Linear(128, 2),
        )
        a = 4
        self.linear_8_to_4 = nn.Sequential(
                nn.Linear(a*a*4, a*a),
                nn.LeakyReLU(opt.lrelu),
                nn.Dropout(opt.dropout),
            )
        a = 8
        self.linear_16_to_8 = nn.Sequential(
                nn.Linear(a*a*4, a*a),
                nn.LeakyReLU(opt.lrelu),
                nn.Dropout(opt.dropout),
            )
        a = 16
        self.linear_32_to_16 = nn.Sequential(
                nn.Linear(a*a*4, a*a),
                nn.LeakyReLU(opt.lrelu),
                nn.Dropout(opt.dropout),
            )
        self.stage = stage
        self.alpha = 0

    def forward(self, img):
        batch_size = img.size(0)
        img = img.view(img.size(0), -1)

        pool = nn.AvgPool2d(2)
        if self.stage == 3:
            img_16 = self.linear_32_to_16(img)
            img_16_16 = pool(img.view(batch_size, 1, 32, 32))
            img = img_16 * self.alpha + img_16_16.view(batch_size, -1) * (1 - self.alpha)
        if self.stage >= 2:
            img_8 = self.linear_16_to_8(img)
            if self.stage >= 2:
                img_8_8 = pool(img.view(batch_size, 1, 16, 16))
                img = img_8 * self.alpha + img_8_8.view(batch_size, -1) * (1 - self.alpha)
            else:
                img = img_8
        if self.stage >= 1:
            img_4 = self.linear_8_to_4(img)
            if self.stage >= 2:
                img_4_4 = pool(img.view(batch_size, 1, 8, 8))
                img = img_4 * self.alpha + img_4_4.view(batch_size, -1) * (1 - self.alpha)
            else:
                img = img_4

        return self.image2score(img)


class Discriminator(nn.Module):

    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.alpha = 0

        self.conv_from_rgb = nn.Conv2d(opt.im_chan, opt.fm_chan, 1)
        blur_f = None

        conv_and_pool = [
            nn.Conv2d(opt.fm_chan, opt.fm_chan, 3, padding=(1, 1)),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            Blur2d(blur_f),
            nn.AvgPool2d(2),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
        ]

        conv_down = [
            nn.Conv2d(opt.fm_chan, opt.fm_chan, 3, padding=(1, 1)),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            Blur2d(blur_f),
            nn.Conv2d(opt.fm_chan, opt.fm_chan, 2, stride=2),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
        ]

        seq = []
        for i in range(3, int(np.log2(opt.im_res))):
            seq += conv_and_pool

        self.convs = nn.Sequential(*seq)

        self.image2score = nn.Sequential(
            nn.Linear(opt.res_0 * opt.res_0 * opt.fm_chan, 128),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, img: Tensor):
        batch_size = img.size(0)
        fmap = self.conv_from_rgb(img)
        fmap = self.convs(fmap)

        return self.image2score(fmap.view(batch_size, -1))