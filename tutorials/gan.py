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
CUDA = False # torch.cuda.is_available()

CLIP_VALUE = 1
CRITIC_ITER = 2

data_loader = DataLoader(
    datasets.MNIST(
        root='data/mnist',
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(200, 784),
            nn.Tanh(),
        )

    def forward(self, z: Tensor):
        z = z.view(z.size(0), -1)
        imgs = self.mlp(z)
        return imgs


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(200, 1),
        )

    def forward(self, imgs: Tensor):
        imgs = imgs.view(imgs.size(0), -1)
        logits = self.mlp(imgs)
        return logits


def compute_gradient_penalty(D, real_samples: Tensor, fake_samples: Tensor):
    alpha = torch.rand(real_samples.size(0), 1)
    interp = (alpha * real_samples + (-alpha + 1) * fake_samples).requires_grad_(True)
    d_interp = D(interp)
    fake_label = torch.ones(real_samples.size(0), 1).requires_grad_(False)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=fake_label,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


generator = Generator()
discriminator = Discriminator()

if CUDA:
    generator.cuda()
    discriminator.cuda()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCH):
    for i, (images, _) in enumerate(data_loader):
        assert isinstance(images, Tensor)
        z = torch.randn(BATCH_SIZE, 784)
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
            g_loss = -torch.mean(fake_logits)
            g_loss.backward()
            g_optimizer.step()

        d_optimizer.zero_grad()
        real_logits = discriminator(images)
        fake_images = generator(z)
        fake_logits = discriminator(fake_images)

        # Adversarial loss
        real_images = images.view(images.size(0), -1)
        gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_images.data)
        d_loss = -torch.mean(real_logits) + torch.mean(fake_logits) + 10 * gradient_penalty
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

    z = torch.randn(BATCH_SIZE, 784)
    if CUDA:
        z = z.cuda()
    imgs = generator(z)
    imgs = imgs.view(imgs.size(0), 1, 28, 28)
    save_image(imgs.data[:25], "save/gan_images/image_%d.png" % epoch, nrow=5, normalize=True)
    torch.save(generator, 'save/gan_model/g')
    torch.save(discriminator, 'save/gan_model/g')

if __name__ == '__main__':
    pass

