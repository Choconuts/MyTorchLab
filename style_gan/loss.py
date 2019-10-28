import torch
import numpy as np
import math
from torch import nn, Tensor
from torch.nn import functional as nnf
from style_gan.config import opt

from style_gan.ops import *


def g_loss_func(fake_validity=None, real_validity=None, real_images=None, fake_images=None, real_label=None, fake_label=None):
    return torch.mean(nnf.softplus(fake_validity))


def d_loss_func(fake_validity=None, real_validity=None, real_images=None, fake_images=None, real_label=None, fake_label=None):
    d_loss_gan = nnf.softplus(fake_validity) + nnf.softplus(-real_validity)
    real_loss = torch.sum(real_validity)
    real_grads = torch.autograd.grad(real_loss, [real_images], retain_graph=True)[0]
    r1_penalty = torch.sum((real_grads ** 2), [1, 2, 3])

    d_loss = d_loss_gan + r1_penalty * (opt.r1_gamma * 0.5)
    return torch.mean(d_loss)


def loss_func(**kwargs):
    return g_loss_func(**kwargs), d_loss_func(**kwargs)