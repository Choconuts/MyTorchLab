from com.learning.ground_truth import *
from com.path_helper import *
import numpy as np, json
from com.mesh.smooth import *


smooth_times = 5
template = Mesh()
betas = np.zeros(0)
zero_shape_index = -1

ex_dir = r'/Users/choconut/Desktop/edu/CAD/project/LBAC/db/raw/pants_mesh/'
cloth_dir = 'shapeModel_lp0 1'
topo_file = 'triangles_clt.txt'

faces = np.zeros(0)


def txt_to_array_3(txt_file, dtype):
    with open(txt_file, 'r') as fp:
        s = ' '
        data = []
        while s:
            s = fp.readline()
            if len(s) < 2:
                continue
            values = s.split(' ')
            tri = []
            for i in range(3):
                vi = values[i]
                tri.append(vi)
            data.append(tri)
        data = np.array(data, dtype)
    return data


def txt_to_array(txt_file, dtype):
    with open(txt_file, 'r') as fp:
        s = None
        data = []
        def read_value():
            s = fp.readline()
            if len(s) < 1:
                return None
            values = s.split(' ')
            return values[-1]

        while s is None:
            s = read_value()
        w = int(float(s))
        s = read_value()
        h = int(float(s))

        data = np.zeros((h, w), dtype)
        for i in range(h):
            for j in range(w):
                s = read_value()
                data[i, j] = s
        return data


def ex_beta_mesh(i):
    cloth_file = 'cloth_' + str3(i) + '.txt'
    cloth_file = join(ex_dir, cloth_dir, cloth_file)
    vertices = txt_to_array_3(cloth_file, 'f')
    global faces
    if faces is None or len(faces) == 0:
        faces = txt_to_array_3(join(ex_dir, topo_file), 'i')
        faces -= 1
    mesh = Mesh().from_vertices(vertices, faces)
    mesh.update()
    return mesh


def smooth_mesh(mesh):
    return smooth(mesh, smooth_times)


class AESampleId(SampleId):
    def derefer(self):
        return [self.data[self.id]]


class AutoEncoderGroundTruth(GroundTruth):

    def load(self, gt_dir):

        self.data = []
        for i in range(17):
            cloth_file = 'cloth_' + str3(i) + '.txt'
            cloth_file = join(ex_dir, cloth_dir, cloth_file)
            vertices = txt_to_array_3(cloth_file, 'f')
            self.data.append(vertices)

        self.samples = []
        for i in range(17):
            sample = AESampleId(i, self.data)
            self.samples.append(sample)

        self.batch_manager = BatchManager(17, 17)

        return self

    def get_batch(self, size):
        ids = self.batch_manager.get_batch(size)
        batch = [[]]
        for id in ids:
            sample = self.samples[id].derefer()
            batch[0].append(sample[0])
        return batch


def set_smooth_times(i):
    global smooth_times
    smooth_times = i



def tst():
    beta_gt = AutoEncoderGroundTruth().load(conf_path('gt/ex/beta/2019-6-19'))

    def pt(i):
        return conf_path('beta_' + str3(i) + '.obj', 'tst')

    print(beta_gt.data)
    # print(beta_gt.template.vertices.__len__() * 3)

    batch = beta_gt.get_batch(4)
    print(np.shape(batch[0]))

    # for i in range(17):
    #     mesh = Mesh(beta_gt.template)
    #     mesh.vertices += beta_gt.data['disps'][str(i)]
    #     mesh.update_normal_only()
    #     mesh.save(pt(i))


if __name__ == '__main__':
    """
    """
    tst()
    # set_smooth_times(0)
    # mesh = ex_beta_mesh(0)
    # smooth_mesh(mesh)
    # mesh.save(conf_path('smooth.obj', 'tst'))
    # load_betas()
    # gen_beta_gt_data(conf_path('gt/ex/beta/2019-6-19'))
