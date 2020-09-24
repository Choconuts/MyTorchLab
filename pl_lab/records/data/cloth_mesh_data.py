import torch
import numpy as np
import math
from torch import nn, Tensor
from torch.nn import functional as nnf
from torch.nn import utils as nnu
from torchvision import datasets
import glob
import random
from pl_lab.records.mesh.mesh import *
from pl_lab.records.mesh.smooth import smooth as smooth_mesh


def read_matrix(file, dtype=float):
    with open(file, 'r') as fp:
        rows = []
        flag = False
        while True:
            line = fp.readline()
            if len(line) <= 1:
                if flag:
                    break
                else:
                    continue
            else:
                flag = True
            nums = line.strip().replace('\t', ' ').split(' ')
            rows.append([dtype(n) for n in nums])
        return np.array(rows)


def write_matrix(mat, file):
    with open(file, 'w') as fp:
        lines = [' '.join(['%.6f' % s for s in row]) + '\n' for row in mat]
        fp.writelines(lines)


class GarmentDataset:

    def __init__(self, folder, transformer=None, tensor=True, device='cpu',
                 dtype=torch.float32, tri_index=1, none_padding=True):
        """

        :param folder: 目标文件夹，包括BetaTraining，cloth_%03d，triangles（optional），会生成一些cache
        :param batch_size: 加载时的批大小
        :param shuffle: 每个迭代epoch会先洗索引
        :param transformer: 对cloth做的变换，一般是Transformer（减去均值后归一化），或者None（不做变换）
        :param tensor: 是否把结果转化成Tensor
        :param device: 前提是转化成Tensor，输出到的设备（cpu， cuda等）
        :param dtype: 转化成的Tensor的dtype，一般就是float32
        :param tri_index: triangles的起始index，一般obj文件读取出来的都是1开始的，设为1
        :param none_padding: 如果beta/cloth中一方sample数量少于另一方，是否用None填充，一般不要让数据里的数量不匹配
        """
        self.folder = folder
        self.is_testing = False
        self.files = glob.glob(r'%s/cloth_*.txt' % (folder,))
        self.cloth_len = len(self.files)
        try:
            self.faces = read_matrix(r'%s/triangles.txt' % folder, int) - tri_index
        except FileNotFoundError as e:
            print(e)
            self.faces = None

        self.betas = read_matrix(folder + '/BetaTraining.txt', float).reshape((-1, 4))
        try:
            self.betas_tests = read_matrix(folder + '/BetaTesting.txt', float).reshape((-1, 4))
        except FileNotFoundError as e:
            print(e)
            self.betas_tests = None
        self.beta_len = self.betas.shape[0]

        self.fun = max if none_padding else min
        self.map = np.linspace(0, self.len - 1, self.len, dtype=int)

        self.transform = transformer if transformer is not None else lambda x: x
        try:
            self.vert_num = self.load_sample(0)[0].shape[0]
        except: self.vert_num = 1
        self.cast = lambda x: x if not tensor else torch.tensor(x, dtype=dtype).to(device)

    @property
    def len(self):
        if self.is_testing:
            return self.betas_tests.shape[0]
        return self.fun(self.beta_len, self.cloth_len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        res = [None, None]
        try:
            res[0] = self.transform(read_matrix(self.files[index], float))
        except IndexError:
            pass
        try:
            if self.is_testing:
                res[1] = self.betas_tests[index]
            else:
                res[1] = self.betas[index]
        except IndexError:
            pass
        res = [self.cast(res[i]) if res[i] is not None else None for i in range(2)]
        return res

    def testing(self):
        self.is_testing = True
        return self

    def save_summary(self, tensor: Tensor, path):
        verts = tensor.view(-1, 3)
        if isinstance(self.transform, Transformer):
            verts = self.transform.invert(verts)
        Mesh().from_vertices(verts, self.faces, no_bounds_info=True).save(path + '.obj')
        write_matrix(verts, path + '.txt')


class Transformer:

    def __init__(self, do_normalize=False):
        self.average = None
        self.avg = 0
        self.std = 1
        self.do_norm = do_normalize

    def __call__(self, cloth):
        assert self.average is not None
        return ((cloth - self.average).reshape(-1) - self.avg) / self.std

    def generate(self, folder):
        garment = GarmentDataset(folder, tensor=False)
        # garment.batch_size = garment.len
        # for batch, _ in garment:
        #     self.average = np.mean(batch, axis=0)
        for verts, beta in garment:
            if (beta == 0).all():
                self.average = verts[0]
        write_matrix(self.average, folder + '/average(generated).txt')
        disp_avg = np.mean(self.average)
        disp_std = np.std(self.average)
        write_matrix([[disp_avg], [disp_std]], folder + '/avg_std(generated).txt')
        return self

    def load(self, folder):
        self.average = read_matrix(folder + '/average(generated).txt')
        self.avg, self.std = read_matrix(folder + '/avg_std(generated).txt').reshape((-1))
        self.avg = 0
        if not self.do_norm:
            self.std = 1
        return self

    def invert(self, displacements: Tensor):
        assert self.average is not None
        return (displacements.cpu().detach().numpy() * self.std + self.avg).reshape(-1, 3) + self.average


def easy_dataset(folder,
                 train,                                     # is for train or test TODO
                 smooth=False,                              # TODO
                 to_disp=True,                              # minus the zero shape positions
                 normalize=False,                           # standard the std-dev
                 tensor=True,                               # make array to tensors
                 device='cpu',                              # to device
                 dtype=torch.float32,                       # float
                 tri_index=1,                               # faces begin index
                 none_padding=True):                        # if not found, return None
    if to_disp:
        transformer = Transformer()
        try:
            transformer.load(folder)
        except FileNotFoundError:
            transformer.generate(folder)
    else:
        transformer = None
    GarmentDataset(folder=folder,
                   transformer=transformer,
                   tensor=tensor,
                   device=device,
                   dtype=dtype,
                   tri_index=tri_index,
                   none_padding=none_padding)


if __name__ == '__main__':
    folder = r'F:\Data\csm_training_1105'
    loader = GarmentDataset(folder, transformer=None)
    for a, b in loader:
        print(a.shape, b.shape)
        loader.save_summary(a, 'tmp/tst')
        break
    # Transformer().generate(folder)