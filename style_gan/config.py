import torch
from utils.option import OptionBase


class GANOption(OptionBase):

    model_path = 'save/model/state.pth'
    data_path = '../tutorials/data/mnist'
    img_path = 'save/images'
    sum_path = 'save/images'

    epoch = 1000
    batch_size = 50
    show_steps = 200
    save_steps = 1

    learning_rate = 1e-5
    dropout = 0.2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ Global """
    im_res = 32
    im_chan = 1
    lrelu = 0.2
    spec_norm = True
    use_ws = True

    """ Generator """
    latent_size = 512
    fc_units = 512
    fm_chan = 512    # feature map channels of the first layer
    res_0 = 4       # start resolution
    st_thr = 1      # stage where the fm_res begin to decline
    noise = 'rand'     # fix, none(/no) and rand
    norm = 'I'      # adain
    const_start = False
    blur = False
    bilinear = True

    """ Discriminator """
    r1_gamma = 10

    """ Train Control """
    clip_value = 0.05
    critic_iter = 6
    genera_iter = 1

    level_up_epoch = -1


opt = GANOption()

""" Test AdaIN """
if True:
    opt.noise = 'fix'
    opt.blur = True
    opt.bilinear = False
    opt.norm = 'A'
    opt.const_start = True


def make_image(tensor: torch.Tensor):
    from math import sqrt
    if len(tensor.shape) == 4:
        return tensor
    if len(tensor.shape) == 2:
        n = tensor.size(1)
        a = int(sqrt(n))
        b = n // a
        tensor = tensor[:, :a * b].view(tensor.size(0), 1, a, b)
        return tensor


def save_sum(imgs: torch.Tensor, id=0):
    from torchvision.utils import save_image
    from os import path
    imgs = make_image(imgs)
    imgs = imgs.data[:25, :3]
    save_image(imgs, path.join(opt.sum_path, "summary_%d.png" % id), nrow=5, normalize=True)

