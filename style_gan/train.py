import numpy as np
import math
from torch.utils.data import DataLoader
from torch import nn, Tensor

from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

from style_gan.config import *
from style_gan.generator import Generator, Generator0
from style_gan.discriminator import Discriminator
from style_gan.loss import *

import os.path as path

# use MNIST for test
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

# set stage to 16 * 16
generator = Generator(2).to(opt.device)
discriminator = Discriminator(2).to(opt.device)
cross_entropy = nn.CrossEntropyLoss().to(opt.device)
mse_loss = nn.MSELoss().to(opt.device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=opt.learning_rate)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate)

# train loop
for epoch in range(opt.epoch):
    for i, (images, _) in enumerate(data_loader):
        assert isinstance(images, Tensor)
        # inputs

        real_label = torch.ones(opt.batch_size).long().to(opt.device)
        fake_label = torch.zeros(opt.batch_size).long().to(opt.device)
        images = images.to(opt.device).requires_grad_(True)

        # run g_optim once after running each X d_optim, X is critic_iter
        if i % opt.critic_iter == 0:
            generator.train()
            z = torch.randn(opt.batch_size, opt.latent_size).to(opt.device)
            g_optimizer.zero_grad()
            fake_images = generator(z)
            fake_validity = discriminator(fake_images)
            g_loss = -torch.mean(fake_validity)
            # g_loss = cross_entropy(fake_validity, real_label)
            # g_loss = mse_loss(fake_validity, real_label.view(-1, 1))
            # g_loss = g_loss_func(fake_validity=fake_validity)
            g_loss.backward()
            g_optimizer.step()

        if i % opt.genera_iter == 0:
            z = torch.randn(opt.batch_size, opt.latent_size).to(opt.device)
            d_optimizer.zero_grad()
            images = nn.functional.interpolate(
                images.view(-1, 1, 32, 32),
                size=(4 * 2 ** generator.stage, 4 * 2 ** generator.stage)
            )

            real_validity = discriminator(images)
            fake_images = generator(z)
            fake_validity = discriminator(fake_images)

            # use cross entropy
            # real_images = images.view(images.size(0), -1)
            # gradient_penalty = compute_gradient_penalty(discriminator, real_images.cuda(), fake_images.cuda())
            # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) # + 10 * gradient_penalty
            # d_loss = cross_entropy(real_validity, real_label) + cross_entropy(fake_validity, fake_label)
            # d_loss = mse_loss(real_validity, real_label.view(-1, 1)) + mse_loss(fake_validity, fake_label.view(-1, 1))
            d_loss = d_loss_func(
                fake_validity,
                real_validity,
                images,
                fake_images,
                real_label,
                fake_label,
            )
            d_loss.backward()
            d_optimizer.step()

            # clip weights of discriminator
            for p in discriminator.parameters():
                p.data.clamp_(-opt.clip_value, opt.clip_value)

        if i % opt.show_steps == 0 and i >= opt.critic_iter and i >= opt.genera_iter:
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