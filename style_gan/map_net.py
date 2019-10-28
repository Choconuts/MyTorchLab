import torch
import numpy as np
import math
from torch import nn, Tensor
from torch.nn import functional as nnf
from torch.nn import utils as nnu

from style_gan.ops import *
from style_gan.config import *


class MapNet(nn.Module):

    def __init__(self, units=512, layers=8, do_norm=True):
        super(MapNet, self).__init__()

        self.mapping = nn.Sequential()
        self.do_norm = do_norm

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
            # nn.Dropout(opt.dropout),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            # nn.Linear(units, units),
            # nn.LeakyReLU(opt.lrelu),

            # nn.Linear(units, units),
            # nn.LeakyReLU(opt.lrelu),
            # nn.Linear(units, units),
            # nn.LeakyReLU(opt.lrelu),
            #
            # nn.Linear(units, units),
            # nn.LeakyReLU(opt.lrelu),
            # nn.Linear(units, opt.latent_size),
            # nn.LeakyReLU(opt.lrelu),
        )

        self.pixel_norm = PixelNorm()

        for m in self.simple_mapping.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, 0.5)

    def forward(self, z: Tensor):
        if self.do_norm:
            z = self.pixel_norm(z)
        # save_sum(z, 1)
        batch_size = z.size(0)
        z = z.view(batch_size, -1)
        w = self.simple_mapping(z)
        # save_sum(w, 2)
        return w


class GMapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 resolution=1024,
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,         # Enable equalized learning rate?
                 lrmul=0.01,              # Learning rate multiplier for the mapping layers.
                 gain=2**(0.5)            # original gain in tensorflow.
                 ):
        super().__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        )

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out #, self.num_layers



