# network imports
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import utils as nnu


class HighParam:
    """ MLP """
    units = [400, 600]
    lrelu = 0.2
    dropout = 0.2
    sigmoid = False


class MLP(nn.Module):

    def __init__(self,
                 input_size,                                        # Mlp input tensor size
                 output_size,                                       # Mlp output tensor size
                 hparam: HighParam = None                           # inner parameters
                 ):
        super().__init__()
        if hparam is None:
            hparam = HighParam()
        self.hparam = hparam
        units = [input_size]
        units.extend(hparam.units)

        self.layers = nn.ModuleList()

        for i in range(1, len(units)):
            self.layers.append(nn.Sequential(
                nn.Linear(units[i - 1], units[i]),
                nn.LeakyReLU(hparam.lrelu),
                nn.Dropout(hparam.dropout),
            ))
        self.final_mlp = nn.Linear(units[-1], output_size)
        self.n_layer = len(units)

    def forward(self, x: Tensor):
        bsize = x.size(0)
        x = x.reshape(bsize, -1)
        for i in range(self.n_layer - 1):
            x = self.layers[i](x)
        x = self.final_mlp(x)
        if self.hparam.sigmoid:
            x = torch.sigmoid(x)
        return x


if __name__ == '__main__':
    mlp = MLP(784, 10)
    print(mlp)
    from torchsummaryX import summary
    summary(mlp, torch.zeros(1, 784))


