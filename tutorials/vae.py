import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn, Tensor

from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from itertools import chain

BATCH_SIZE = 80
LEARNING_RATE = 1e-3
EPOCH = 1000
SHOW_STEPS = 200
SAVE_STEP = 1
CODE_SIZE = 200
DROP_RATE = 0.2
CUDA = torch.cuda.is_available()


data_loader = DataLoader(
    datasets.MNIST(
        root='data/mnist',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        n_input = 576
        self.linear = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROP_RATE),
            nn.Linear(512, CODE_SIZE),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROP_RATE),
            nn.Linear(512, CODE_SIZE),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROP_RATE),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2),
            nn.Dropout(DROP_RATE),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, 2),
            nn.Dropout(DROP_RATE),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, 1),
            nn.Dropout(DROP_RATE),
            nn.LeakyReLU(0.2),
        )

    def forward(self, images: Tensor):
        batch_size = images.shape[0]
        images = images.view(batch_size, -1)
        side = int(images.size(1)**0.5)
        images = images.view(batch_size, 1, side, side)

        # x = nn.Conv2d(1, 64, 4, 2)(images)
        # x1 = nn.Conv2d(64, 64, 4, 2)(x)
        # x2 = nn.Conv2d(64, 64, 4, 1)(x1)
        conv_img = self.conv2d(images)
        conv_img = conv_img.view(batch_size, -1)
        mean = self.linear(conv_img)
        logd = self.linear2(conv_img)
        return add_dev(mean, torch.exp(logd)), mean, logd


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(CODE_SIZE, 200),
            nn.LeakyReLU(0.2),
            nn.Linear(200, 32 * 32),
            nn.LeakyReLU(0.2),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, 1),
        )

        self.dense = nn.Sequential(
            nn.Linear(CODE_SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 64 * 12 * 12),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2),
            nn.Dropout(DROP_RATE),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 4, 1),
            nn.Dropout(DROP_RATE),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 1, 4, 1),
            nn.Dropout(DROP_RATE),
            nn.LeakyReLU(0.2),
        )

    def forward(self, z: Tensor):
        batch_size = z.size(0)
        z = z.view(batch_size, -1)
        # img = self.linear(z)
        img = self.dense(z)
        side = int((img.size(1) // 64)**0.5)
        # out = self.conv(img.view(batch_size, 1, side, side))
        out = self.deconv(img.view(batch_size, 64, side, side))
        # out = out.view(batch_size, -1)
        return torch.tanh(out)


def add_dev(code: Tensor, dev: Tensor):
    if CUDA:
        return code + dev * torch.randn(code.size()).cuda()
    return code + dev * torch.randn(code.size())


encoder = Encoder()
decoder = Decoder()
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(chain(encoder.parameters(), decoder.parameters()), lr=LEARNING_RATE)

if CUDA:
    encoder.cuda()
    decoder.cuda()
    mse_loss.cuda()


for epoch in range(EPOCH):
    for i, (images, _) in enumerate(data_loader):
        assert isinstance(images, Tensor)
        encoder.train()
        decoder.train()

        if CUDA:
            images = images.cuda()

        optimizer.zero_grad()
        z, mean, logd = encoder(images)
        # print(z)
        rebuild = decoder(z)
        KL = -0.5 * torch.sum(1 + 2. * logd - mean.pow(2) - torch.exp(2. * logd), [1])
        loss = mse_loss(images, rebuild) #pass+ torch.mean(KL)
        loss.backward()
        optimizer.step()

        if i % SHOW_STEPS == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f]"
                % (epoch, EPOCH, i, len(data_loader), loss.item())
            )

    if epoch % SAVE_STEP == 0 and epoch > 0:
        encoder.eval()
        decoder.eval()
        z = torch.randn(BATCH_SIZE, CODE_SIZE)
        if CUDA:
            z = z.cuda()
        imgs = decoder(z)
        # imgs = rebuild
        # imgs = imgs.view(imgs.size(0), 1, int(np.sqrt(imgs.size(1))), int(np.sqrt(imgs.size(1))))
        save_image(imgs.data[:25], "save/vae_images/image_%d.png" % epoch, nrow=5, normalize=True)
        torch.save(decoder, 'save/vae_model/d-%d' % epoch)
        torch.save(encoder, 'save/vae_model/c-%d' % epoch)


if __name__ == '__main__':
    pass