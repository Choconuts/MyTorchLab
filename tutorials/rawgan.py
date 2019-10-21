import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn, Tensor

from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

BATCH_SIZE = 80
LEARNING_RATE = 1e-3
EPOCH = 100
SHOW_STEPS = 200
CODE_SIZE = 100
CUDA = False # torch.cuda.is_available()

CLIP_VALUE = 0.01
CRITIC_ITER = 2

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


class Generator(nn.Module):

    def __init__(self, level, units):
        super(Generator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(level*level, units),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(units, level*level),
            nn.Tanh(),
        )

    def forward(self, z: Tensor):
        shape = z.shape
        z = z.view(z.size(0), -1)
        y = self.mlp(z)
        imgs = y.view(shape)
        return imgs


class Discriminator(nn.Module):

    def __init__(self, level, units):
        super(Discriminator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(level*level, units),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(units, 2),
        )

    def forward(self, imgs: Tensor):
        imgs = imgs.view(imgs.size(0), -1)
        logits = self.mlp(imgs)
        return logits


generator = Generator(32, 200)
discriminator = Discriminator(32, 200)
cross_entropy = nn.CrossEntropyLoss()

if CUDA:
    generator.cuda()
    discriminator.cuda()
    cross_entropy.cuda()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH):
    for i, (images, _) in enumerate(data_loader):
        assert isinstance(images, Tensor)
        z = torch.randn(BATCH_SIZE, CODE_SIZE)
        real_label = torch.ones(BATCH_SIZE).long()
        fake_label = torch.zeros(BATCH_SIZE).long()
        if CUDA:
            images = images.cuda()
            z = z.cuda()
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()

        if i % CRITIC_ITER == 0:
            g_optimizer.zero_grad()
            fake_logits = discriminator(generator(z))
            g_loss = cross_entropy(fake_logits, real_label)
            g_loss.backward()
            g_optimizer.step()

        d_optimizer.zero_grad()
        real_logits = discriminator(images)
        fake_logits = discriminator(generator(z))

        d_loss = cross_entropy(fake_logits, fake_label) + cross_entropy(real_logits, real_label)
        d_loss.backward()
        d_optimizer.step()

        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

        if i % SHOW_STEPS == 0 and i % CRITIC_ITER == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f]"
                % (epoch, EPOCH, i, len(data_loader), g_loss.item(), d_loss.item())
            )

    z = torch.randn(BATCH_SIZE, CODE_SIZE)
    if CUDA:
        z = z.cuda()
    imgs = generator(z)
    imgs = imgs.view(imgs.size(0), 1, 32, 32)
    save_image(imgs.data[:25], "save/gan_images/image_%d.png" % epoch, nrow=5, normalize=True)


if __name__ == '__main__':
    pass

