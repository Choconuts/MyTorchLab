from multiprocessing import Process
import itertools
from collections.abc import Iterable
import os, sys
import numpy as np

train = 'gan.py'
cmd = 'python3 %s ' % train
opts = {
    ''
}

PYTHON = r'C:\Users\MSI\AppData\Local\Programs\Python\Python37\python.exe '
TARGET = r'E:/PycharmProjects/TorchLab/style_gan/train.py '

options = {
    'epoch': 300,
    'batch_size': 50,
}

condition = {

}
variation = {
    'critic_iter': range(1, 6),
    'norm': ['AA', 'IR'],
    'epoch': np.linspace(100, 300, 2)
}


def run(img_path, model_path, critic):
    os.system(PYTHON + ' ' + TARGET + ' --img_path ' + img_path + ' --model_path ' + model_path
              + ' --data_path ' + '../data/mnist' + ' --critic_iter ' + str(critic))


def run_script(target, python=PYTHON, option_dict: dict = None):
    cmd = python + ' ' + target + ' '
    for k, v in option_dict.items():
        cmd += ' --' + k + ' ' + str(v)
    os.system(cmd)






if __name__ == '__main__':
    # p1 = Process(target=run, args=('../../style_gan/save/images_1', '../../style_gan/save/model/state-1.pth', 5))
    # p2 = Process(target=run, args=('../../style_gan/save/images_2', '../../style_gan/save/model/state-2.pth', 1))
    # p1.start()
    # p2.start()
    pass
