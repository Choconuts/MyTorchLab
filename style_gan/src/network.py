import torch
import numpy as np
import math
from torch import nn, Tensor
from torch.nn import functional as nnf
from torch.nn import utils as nnu
from utils import *
from utils.config import *
from style_gan.ops import *


class Config(ConfigBase):
    im_res = 128
    im_chan = 1
    lrelu = 0.2
    dropout = 0.2
    spec_norm = True
    use_ws = True
    latent_size = 512
    fc_units = 512
    fm_chan = 512    # feature map channels of the first layer
    res_0 = 4       # start resolution
    st_thr = 3      # stage where the fm_res begin to decline
    noise = 'fix'     # fix, none(/no) and rand
    norm = 'I'      # adain
    const_start = False
    blur = True
    bilinear = False
    start_alpha = 1.
    r1_gamma = 10


opt = Config()


class GLayer(nn.Module):

    def __init__(self,
                 channel,
                 epsilon=1e-8,
                 no_noise=None,             # None for auto
                 no_bias=False,             # use zero bias
                 use_norm='O'               # N for None, I for Instance, P for Pixel and A for AdaIN, O for auto
                 ):
        super().__init__()
        if use_norm == 'O':
            use_norm = opt.norm
        if no_noise is None:
            no_noise = opt.noise in ['no', 'not', 'none']
        self.no_noise = no_noise
        self.no_bias = no_bias
        self.use_pixel_norm = use_norm == 'P'
        self.use_instance_norm = use_norm == 'A' or use_norm == 'I'
        self.use_style = use_norm == 'A'

        """ Add Noise """
        if not self.no_noise:
            self.noise = ApplyScaledNoise(channel)

        """ Apply Bias """
        if not no_bias:
            self.bias = nn.Parameter(torch.zeros(channel), True)

        """ Pixel Norm """
        if self.use_pixel_norm:
            # self.pixel_norm = nn.LocalResponseNorm(1)
            self.pixel_norm = PixelNorm(epsilon)

        """ Instance Norm """
        if self.use_instance_norm:
            # self.instance_norm = nn.InstanceNorm2d(out_channel, epsilon, affine=False)
            self.instance_norm = InstanceNorm(epsilon)


        """ Apply Style """
        if self.use_style:
            self.style = ApplyStyleAffine(opt.latent_size, channel, opt.use_ws)

    def forward(self, img: Tensor, w_latent: Tensor = None, noise: Tensor = None):
        # add noise
        x = img if self.noise is None else self.noise(img, noise)
        # apply bias
        x = x if self.bias is None else x + self.bias.view(1, -1, 1, 1)
        # activation
        x = nnf.leaky_relu(x, negative_slope=opt.lrelu)
        # normalization
        x = x if self.use_pixel_norm else self.instance_norm(x)
        x = x if self.use_instance_norm else self.pixel_norm(x)
        # apply style
        x = x if not self.use_style else self.style(x, w_latent)
        x = nnf.leaky_relu(x, negative_slope=opt.lrelu)
        return x


class GBlock(nn.Module):

    def __init__(self,
                 stage,                                 # 0 : 4x4 ... 6: 256*256  ... 8: 1024*1024
                 in_channel,                            # last block output fmap channels
                 out_channel,                           # output fmap channels
                 up_func='conv',                        # 'blinear' only for in-chan == out-chan; conv means fused scale
                 ):
        super().__init__()
        self.stage = stage
        self.fm0 = in_channel
        self.fm = out_channel
        self.up_func = up_func

        no_noise = opt.noise in ['no', 'not', 'none']

        if self.up_func == 'bilinear':
            # upsample method 1
            self.conv_up = nn.Sequential(
                Upscale2d(2),
                nn.Conv2d(self.fm0, self.fm, 3, padding=1),
            )

        elif self.up_func == 'conv':
            # upsample method 2
            self.conv_up = nn.ConvTranspose2d(self.fm0, self.fm, 4, stride=2, padding=1)

        self.blur = Blur2d()

        self.layer_0 = GLayer(
            channel=self.fm,
            no_noise=no_noise,
        )

        self.conv = nn.Conv2d(out_channel, out_channel, 3, padding=(1, 1))
        if opt.spec_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        self.layer_1 = GLayer(
            channel=self.fm,
            no_noise=no_noise,
        )

        if opt.noise == 'fix':
            self.noise = torch.randn(2, 1, 1, opt.res_0 * 2 ** self.stage, opt.res_0 * 2 ** self.stage)

    def forward(self, img: Tensor, w_latent: Tensor = None):
        batch_size = img.size(0)
        if opt.noise == 'fix':
            noises = self.noise
        else:
            noises = [None, None]

        # up scale
        if self.stage == 0:
            if not opt.const_start:
                img = self.linear(img.view(batch_size, -1)).view(batch_size, self.fm, 4, 4)
        else:
            img = self.conv_up(img)

        # blur
        if opt.blur:
            img = self.blur(img)
        # epilogue
        img = self.layer_0(img, w_latent, noises[0])
        # conv
        img = self.conv(img)
        # epilogue
        img = self.layer_1(img, w_latent, noises[1])
        return img


class MapNet(nn.Module):

    def __init__(self, units=512, layers=8, do_norm=True):
        super(MapNet, self).__init__()

        self.do_norm = do_norm

        if do_norm:
            self.pixel_norm = PixelNorm()

        self.simple_mapping = nn.Sequential(
            nn.Linear(opt.latent_size, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            nn.Linear(units, units),
            nn.LeakyReLU(opt.lrelu),
            nn.Linear(units, opt.latent_size),
            nn.LeakyReLU(opt.lrelu),
        )

        if opt.spec_norm:
            for m in self.simple_mapping.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight, 0.5)

    def forward(self, z: Tensor):
        if self.do_norm:
            z = self.pixel_norm(z)
        batch_size = z.size(0)
        z = z.view(batch_size, -1)
        w = self.simple_mapping(z)
        return w


class ToRGB(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.conv = Conv2d(channel, opt.im_chan, 1)
        self.bias = nn.Parameter(torch.zeros(opt.im_res), True)

    def forward(self, img):
        return self.conv(img) + self.bias.view(1, -1, 1, 1)


class Generator(nn.Module):

    def __init__(self, stage, alpha=1.):
        super().__init__()
        self.stage = stage      # 0, 1, ..., 8
        self.alpha = alpha
        self.map_net = MapNet(opt.fc_units)
        self.nf = lambda st: int(min(opt.fm_chan, opt.fm_chan / 2 ** (st - opt.st_thr)))

        """
        9 blocks (1 start block and 8 common blocks
                  1 no-up, 3 use up sampling and 5 use conv up)
        18 layers (1 no conv, 17 use conv?)
        9 to RGB convs
        18 noises (9 different sizes, each for 2)
        """
        if opt.const_start:
            self.const_start = nn.Parameter(torch.zeros(1, opt.fm_chan, 4, 4), True)
        else:
            self.linear = nn.Linear(opt.latent_size, opt.res_0*opt.res_0*opt.fm_chan)
        self.layer0 = GLayer(opt.fm_chan)
        self.noise0 = torch.randn(2, 1, 1, opt.res_0, opt.res_0)
        self.conv0 = nn.Conv2d(opt.fm_chan, opt.fm_chan, 3, padding=(1, 1))

        def set_block(i):
            self.__setattr__(
                'block_%d' % i,
                GBlock(
                    stage=i,
                    in_channel=self.nf(i - 1),
                    out_channel=self.nf(i),
                    up_func='bilinear' if i < opt.st_thr else 'conv',
                )
            )

        def set_torgb(i):
            self.__setattr__(
                'torgb_%d' % i,
                ToRGB(
                    channel=self.nf(i),
                )
            )

        self.get_block = lambda i: getattr(self, 'block_%d' % i)
        self.get_torgb = lambda i: getattr(self, 'torgb_%d' % i)
        for i in range(1, 9):
            set_block(i)
        for i in range(0, 9):
            set_torgb(i)

    def forward(self, z: Tensor):
        bsize = z.size(0)
        # Mapping
        if opt.norm == 'A':
            wlatent = self.map_net(z)
            wlatent = wlatent.view(bsize, 1, -1).expand(bsize, 18, opt.latent_size)
        else:
            wlatent = [None] * 18

        # Start
        if opt.const_start:
            start = self.const_start.expand(bsize, opt.fm_chan, 4, 4)
        else:
            start = self.linear(z.view(bsize, -1)).view(bsize, opt.fm_chan, opt.res_0, opt.res_0)

        # First Block
        lst = None
        nxt = self.layer0(start, wlatent[0], self.noise0[0])
        nxt = self.conv0(nxt)
        nxt = self.layer0(nxt, wlatent[1], self.noise0[1])

        # Rest Blocks
        for i in range(1, self.stage + 1):
            tmp = nxt
            nxt = self.get_block(i)(nxt, wlatent[2*i])
            lst = tmp

        # torgb
        out = self.get_torgb(self.stage)(nxt)

        # mix with last state
        if self.stage >= 1:
            out0 = nnf.interpolate(self.get_torgb(self.stage - 1)(lst), scale_factor=2)
            out = out0 * (1 - self.alpha) + out * self.alpha

        return out # torch.tanh(out)


class Discriminator(nn.Module):

    def __init__(self, stage):
        super().__init__()
        self.stage = stage                  # 0, 1, ..., 8
        self.alpha = 1
        fm_chan = opt.fm_chan // 2
        self.conv_from_rgb = nn.Conv2d(opt.im_chan, fm_chan, 1)
        blur_f = None

        conv_and_pool = [
            nn.Conv2d(fm_chan, fm_chan, 3, padding=(1, 1)),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            Blur2d(blur_f),
            nn.AvgPool2d(2),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
        ]

        conv_down = [
            nn.Conv2d(fm_chan, fm_chan, 3, padding=(1, 1)),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            Blur2d(blur_f),
            nn.Conv2d(fm_chan, fm_chan, 2, stride=2),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
        ]

        seq = []
        for i in range(self.stage):
            seq += conv_and_pool

        self.convs = nn.Sequential(*seq)

        self.image2score = nn.Sequential(
            nn.Linear(opt.res_0 * opt.res_0 * fm_chan, 128),
            nn.LeakyReLU(opt.lrelu),
            nn.Dropout(opt.dropout),
            nn.Linear(128, 1),
        )

    def forward(self, img: Tensor):
        batch_size = img.size(0)
        fmap = self.conv_from_rgb(img)
        fmap = self.convs(fmap)

        return self.image2score(fmap.view(batch_size, -1))


if __name__ == '__main__':
    generator = Generator(8)
    discriminator = Discriminator(8)
    print(generator)
    print(discriminator)
    import torchsummaryX as sumx
    sumx.summary(generator, torch.zeros(1, opt.latent_size))
    sumx.summary(discriminator, torch.zeros(1, opt.im_chan, 1024, 1024))
