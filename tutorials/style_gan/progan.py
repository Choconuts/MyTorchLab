import numpy as np
import math
from torch.utils.data import DataLoader
from torch import nn, Tensor

from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from tutorials.style_gan.config import *
from tutorials.style_gan.net import Generator, Generator0, Generator_DC

import os.path as path

data_loader = DataLoader(
    datasets.MNIST(
        root=opt.data_path,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.im_res), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)


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
        self.stage = 0
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
        # if self.level >= 1:
        #     img_4 = self.linear_8_to_4(img)
        #     if self.level >= 2:
        #         img_4_4 = pool(img.view(batch_size, 1, 8, 8))
        #         img = img_4 * self.alpha + img_4_4.view(batch_size, -1) * (1 - self.alpha)
        #     else:
        #         img = img_4

        return self.image2score(img)


generator = Generator(2)
discriminator = Discriminator()
cross_entropy = nn.CrossEntropyLoss()
mse_loss = nn.MSELoss()
discriminator.stage = 2

g_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.learning_rate)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate)


generator.to(opt.device)
discriminator.to(opt.device)
cross_entropy.to(opt.device)
mse_loss.to(opt.device)


def compute_gradient_penalty(D, real_samples: Tensor, fake_samples: Tensor):
    alpha = torch.rand(real_samples.size(0), 1).to(opt.device)
    interp = (alpha * real_samples + (-alpha + 1) * fake_samples).requires_grad_(True)
    d_interp = D(interp)
    fake_label = torch.ones(real_samples.size(0), 1).requires_grad_(False).to(opt.device)

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


for epoch in range(opt.epoch):
    for i, (images, _) in enumerate(data_loader):
        assert isinstance(images, Tensor)
        z = torch.randn(opt.batch_size, opt.latent_size).to(opt.device)
        real_label = torch.ones(opt.batch_size).long().to(opt.device)
        fake_label = torch.zeros(opt.batch_size).long().to(opt.device)
        images = images.to(opt.device)

        if i % opt.critic_iter == 0:
            g_optimizer.zero_grad()
            fake_images = generator(z)
            fake_validity = discriminator(fake_images)
            # g_loss = -torch.mean(fake_validity)
            g_loss = cross_entropy(fake_validity, real_label)
            # g_loss = mse_loss(fake_validity, real_label.view(-1, 1))
            g_loss.backward()
            g_optimizer.step()

        d_optimizer.zero_grad()
        images = nn.functional.interpolate(images.view(-1, 1, 32, 32), size=(4 * 2 ** generator.stage, 4 * 2 ** generator.stage)).view(opt.batch_size, -1)
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
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        if i % opt.show_steps == 0 and i % opt.critic_iter == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f]"
                % (epoch, opt.epoch, i, len(data_loader), g_loss.item(), d_loss.item())
            )

    if epoch % opt.save_steps == 0:
        z = torch.randn(opt.batch_size, opt.latent_size).to(opt.device)
        generator.eval()
        imgs = generator(z)
        # imgs = real_images
        save_image(imgs.data[:25], path.join(opt.img_path, "image_%d.png" % epoch), nrow=5, normalize=True)
        generator.train()
        state = {
            'G': generator.state_dict(),
            'D': discriminator.state_dict(),
            'start_epoch': epoch,
        }
        torch.save(state, opt.model_path)

    if epoch % opt.level_up_epoch == opt.level_up_epoch - 1:
        if generator.stage < 3:
            generator.stage += 1
            discriminator.stage += 1
            generator.alpha = 0
            discriminator.alpha = 0
    if generator.alpha < 1:
        generator.alpha += 0.01
        discriminator.alpha += 0.01

if __name__ == '__main__':
    pass