import numpy as np
import math
import torch
from torch.utils.data import DataLoader
from torch import nn, Tensor

from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image
import os.path as path
import torchsummaryX as sumx
from utils.config import *
from style_gan.src import network
from style_gan.src.optimize import *


@config_class
class Config(ConfigBase):
    epoch = 1000
    batch_size = 8
    show_steps = 200
    save_steps = 1
    epoch_batch = 1600

    learning_rate = 1e-4
    dropout = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    clip_value = 0.5
    critic_iter = 2
    genera_iter = 1

    level_up_epoch = -1

    img_path = '../save/images'
    model_path = '../save/model/src-2.pth'
    resume = False


train_config = Config()
net_config = network.Config(
    im_res=128,
    im_chan=1,
    lrelu=0.2,
    dropout=0.2,
    spec_norm=True,
    use_ws=True,
    latent_size=512,
    fc_units=512,
    fm_chan=512,  # feature map channels of the first layer
    res_0=4,  # start resolution
    st_thr=3,  # stage where the fm_res begin to decline
    noise='fix',  # fix, none(/no) and rand
    const_start=True,
    blur=True,
    bilinear=False,
    start_alpha=1.,
    r1_gamma=10,
    max_stage=6,
    final_tanh=False,
)

# ConfigBase.parse()

# use MNIST for test
data_loader = DataLoader(
    datasets.MNIST(
        root='../../tutorials/data/mnist',
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.Resize(net_config.im_res), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=train_config.batch_size,
    shuffle=True,
)

Generator, Discriminator = network.interface(net_config)

intlog = lambda x: int(round(np.log2(x)))
stage = intlog(net_config.im_res) - 2

generator = Generator().to(train_config.device)
generator.alpha = net_config.start_alpha
sumx.summary(generator, torch.zeros(train_config.batch_size, net_config.latent_size).to(train_config.device), stage)
discriminator = Discriminator().to(train_config.device)
cross_entropy = nn.CrossEntropyLoss().to(train_config.device)
mse_loss = nn.MSELoss().to(train_config.device)

g_optimizer = torch.optim.Adam(generator.parameters(), lr=train_config.learning_rate)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=train_config.learning_rate)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(g_optimizer, [20, 30, 40, 50], 0.1, last_epoch=10)
print('-----------------Train start--------------------')

import os
if not os.path.exists(train_config.img_path):
    os.makedirs(train_config.img_path)
if not os.path.exists(os.path.dirname(train_config.model_path)):
    os.makedirs(os.path.dirname(train_config.model_path))

start_epoch = 0
if train_config.resume:
    if os.path.exists(train_config.model_path):
        state = torch.load(train_config.model_path)
        start_epoch = state['start_epoch'] + 1
        generator.load_state_dict(state['G'])
        discriminator.load_state_dict(state['D'])


# train loop
for epoch in range(start_epoch, train_config.epoch):
    for i, (images, _) in enumerate(data_loader):
        if i > train_config.epoch_batch > 0:
            break
        assert isinstance(images, Tensor)
        # inputs

        real_label = torch.ones(train_config.batch_size).long().to(train_config.device)
        fake_label = torch.zeros(train_config.batch_size).long().to(train_config.device)
        images = images.to(train_config.device).requires_grad_(True)

        # run g_optim once after running each X d_optim, X is critic_iter
        if i % train_config.critic_iter == 0:
            generator.train()
            z = torch.randn(train_config.batch_size, net_config.latent_size).to(train_config.device)
            g_optimizer.zero_grad()
            fake_images = generator(z, stage)
            fake_validity = discriminator(fake_images, stage)
            g_loss = -torch.mean(fake_validity)
            # g_loss = cross_entropy(fake_validity, real_label)
            # g_loss = mse_loss(fake_validity, real_label.view(-1, 1))
            # g_loss = g_loss_func(fake_validity=fake_validity)
            g_loss.backward()
            g_optimizer.step()

        if i % train_config.genera_iter == 0:
            z = torch.randn(train_config.batch_size, net_config.latent_size).to(train_config.device)
            d_optimizer.zero_grad()
            images = nn.functional.interpolate(
                images.view(-1, 1, net_config.im_res, net_config.im_res),
                size=(4 * 2 ** stage, 4 * 2 ** stage)
            )

            real_validity = discriminator(images, stage)
            fake_images = generator(z, stage)
            fake_validity = discriminator(fake_images, stage)

            # use cross entropy
            # real_images = images.view(images.size(0), -1)
            # gradient_penalty = compute_gradient_penalty(discriminator, real_images.cuda(), fake_images.cuda())
            # d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) # + 10 * gradient_penalty
            # d_loss = cross_entropy(real_validity, real_label) + cross_entropy(fake_validity, fake_label)
            # d_loss = mse_loss(real_validity, real_label.view(-1, 1)) + mse_loss(fake_validity, fake_label.view(-1, 1))
            d_loss = calc_d_loss(
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
                p.data.clamp_(-train_config.clip_value, train_config.clip_value)

        if i % train_config.show_steps == 0 and i >= train_config.critic_iter and i >= train_config.genera_iter:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [G loss: %f] [D loss: %f]"
                % (epoch, train_config.epoch, i, len(data_loader), g_loss.item(), d_loss.item())
            )

    if epoch % train_config.save_steps == 0:
        z = torch.randn(train_config.batch_size, net_config.latent_size).to(train_config.device)
        generator.eval()
        imgs = generator(z, stage)
        # imgs = real_images
        save_image(imgs.data[:25], path.join(train_config.img_path, "image_%d.png" % epoch), nrow=5, normalize=True)
        state = {
            'G': generator.state_dict(),
            'D': discriminator.state_dict(),
            'start_epoch': epoch,
        }
        torch.save(state, train_config.model_path)

    if epoch % train_config.level_up_epoch == train_config.level_up_epoch - 1:
        if generator.stage < 3:
            generator.stage += 1
            discriminator.stage += 1
            generator.alpha = 0
            discriminator.alpha = 0
    if generator.alpha < 1 and train_config.level_up_epoch > 0:
        generator.alpha += 0.01
        discriminator.alpha += 0.01


if __name__ == '__main__':
    pass