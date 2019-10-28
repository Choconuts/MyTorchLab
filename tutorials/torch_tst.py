import torch
import numpy as np
import math
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn, Tensor
from torch.nn import functional as nnf

from torchvision import datasets
from torchvision.transforms import transforms
from torchvision.utils import save_image

a = torch.ones(10, 3, 4, 5)
b = torch.ones(10, 3) * 0.5

r = a.permute([2, 3, 0, 1]) * b
print(r.permute([2, 3, 0, 1]).shape)

if __name__ == '__main__':
    pass