import torch
from utils.option import OptionBase


class GANOption(OptionBase):

    model_path = 'save/gan_model/state-2.pth'
    data_path = '../data/mnist'
    img_path = 'save/style_gan_images'

    epoch = 1000
    batch_size = 50
    show_steps = 200
    save_steps = 1

    learning_rate = 1e-3
    dropout = 0.2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    """ Global """
    im_res = 32
    im_chan = 1
    lrelu = 0.2

    """ Generator """
    latent_size = 512
    fc_units = 512
    fm_chan = 512    # feature map channels of the first layer
    st_thr = 1      # stage where the fm_res begin to decline

    """ Discriminator """

    """ Train Control """
    clip_value = 0.05
    critic_iter = 2

    level_up_epoch = 300


opt = GANOption()