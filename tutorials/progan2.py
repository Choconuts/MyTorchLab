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
EPOCH = 4000
SHOW_STEPS = 200
SAVE_STEP = 100
CODE_SIZE = 200
CUDA = torch.cuda.is_available()

CLIP_VALUE = 0.05
CRITIC_ITER = 2

LEVEL_UP_EPOCH = 1000

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

    def __init__(self):
        super(Generator, self).__init__()

        self.code2image = nn.Sequential(
            nn.Linear(CODE_SIZE, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        a = 8
        self.linear_8_to_16 = nn.Sequential(
            nn.Linear(a*a, a*a*4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
        )
        a = 16
        self.linear_16_to_32 = nn.Sequential(
                nn.Linear(a*a, a*a*4),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            )
        self.conv_2d = nn.Conv2d(1, 1, 1)
        self.level = 0
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
        if self.level == 1:
            return torch.tanh(img_8)
        img_16 = self.linear_8_to_16(img_8)
        img_16 = self.conv(img_16)
        if self.level == 2:
            img_8_8_ = img_8.view(batch_size, 1, 8, 8)
            img_16_16 = nn.functional.interpolate(img_8_8_, scale_factor=2)
            combine = img_16_16.view(batch_size, -1) * (1 - self.alpha) + img_16 * self.alpha
            return torch.tanh(combine)
        img_32 = self.linear_16_to_32(img_16)
        img_32 = self.conv(img_32)
        if self.level == 3:
            img_16_16_ = img_16.view(batch_size, 1, 16, 16)
            img_32_32 = nn.functional.interpolate(img_16_16_, scale_factor=2)
            combine = img_32_32.view(batch_size, -1) * (1 - self.alpha) + img_32 * self.alpha
            return torch.tanh(combine)


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.image2score = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )
        a = 4
        self.linear_8_to_4 = nn.Sequential(
                nn.Linear(a*a*4, a*a),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            )
        a = 8
        self.linear_16_to_8 = nn.Sequential(
                nn.Linear(a*a*4, a*a),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            )
        a = 16
        self.linear_32_to_16 = nn.Sequential(
                nn.Linear(a*a*4, a*a),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            )
        self.level = 0
        self.alpha = 0

    def forward(self, img):
        batch_size = img.size(0)
        img = img.view(img.size(0), -1)

        pool = nn.AvgPool2d(2)
        if self.level == 3:
            img_16 = self.linear_32_to_16(img)
            img_16_16 = pool(img.view(batch_size, 1, 32, 32))
            img = img_16 * self.alpha + img_16_16.view(batch_size, -1) * (1 - self.alpha)
        if self.level >= 2:
            img_8 = self.linear_16_to_8(img)
            if self.level >= 2:
                img_8_8 = pool(img.view(batch_size, 1, 16, 16))
                img = img_8 * self.alpha + img_8_8.view(batch_size, -1) * (1 - self.alpha)
            else:
                img = img_8
        # if self.level >= 1:
        #     img_4 = self.linear_8_to_4(img)
        #     if self.level >= 2:
        #         img_4_4 = pool(img.view(batch_size, 1, 8, 8))
        #         img = img_4 * self.alpha + img_4_4.view(batch_size, -1) * (1 - self.alpha)
        #     else:
        #         img = img_4

        return self.image2score(img)


generator = Generator()
discriminator = Discriminator()
cross_entropy = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
generator.level = 1
discriminator.level = 1

g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

if CUDA:
    generator.cuda()
    discriminator.cuda()
    cross_entropy.cuda()
    mse_loss.cuda()


def compute_gradient_penalty(D, real_samples: Tensor, fake_samples: Tensor):
    alpha = torch.rand(real_samples.size(0), 1)
    if CUDA:
        alpha = alpha.cuda()
    interp = (alpha * real_samples + (-alpha + 1) * fake_samples).requires_grad_(True)
    d_interp = D(interp)
    fake_label = torch.ones(real_samples.size(0), 1).requires_grad_(False)
    if CUDA:
        fake_label = fake_label.cuda()
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
            fake_validity = discriminator(generator(z))
            # g_loss = -torch.mean(fake_validity)
            g_loss = cross_entropy(fake_validity, real_label)
            # g_loss = mse_loss(fake_validity, real_label.view(-1, 1))
            g_loss.backward()
            g_optimizer.step()

        d_optimizer.zero_grad()
        images = nn.functional.interpolate(images.view(-1, 1, 32, 32), size=(4 * 2 ** generator.level, 4 * 2 ** generator.level)).view(BATCH_SIZE, -1)
        real_validity = discriminator(images)
        fake_images = generator(z)
        fake_validity = discriminator(fake_images)

        # real_images = images.view(images.size(0), -1)
        # gradient_penalty = compute_gradient_penalty(discriminator, real_images.cuda(), fake_images.cuda())
        # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) # + 10 * gradient_penalty
        d_loss = cross_entropy(real_validity, real_label) + cross_entropy(fake_validity, fake_label)
        # d_loss = mse_loss(real_validity, real_label.view(-1, 1)) + mse_loss(fake_validity, fake_label.view(-1, 1))
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

    if epoch % SAVE_STEP == 0 and epoch > 0:
        z = torch.randn(BATCH_SIZE, CODE_SIZE)
        if CUDA:
            z = z.cuda()
        generator.eval()
        imgs = generator(z)
        # imgs = real_images
        imgs = imgs.view(imgs.size(0), 1, int(np.sqrt(imgs.size(1))), int(np.sqrt(imgs.size(1))))
        save_image(imgs.data[:25], "save/gan_images_final/image_%d.png" % epoch, nrow=5, normalize=True)
        generator.train()
        torch.save(generator, 'save/gan_model/g-%d' % epoch)
        torch.save(discriminator, 'save/gan_model/d-%d' % epoch)

    if epoch % LEVEL_UP_EPOCH == LEVEL_UP_EPOCH - 1:
        if generator.level < 3:
            generator.level += 1
            discriminator.level += 1
            generator.alpha = 0
            discriminator.alpha = 0
    if generator.alpha < 1:
        generator.alpha += 0.01
        discriminator.alpha += 0.01

if __name__ == '__main__':
    pass