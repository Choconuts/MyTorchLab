import numpy as np
from com.mesh.mesh import Mesh, OBJ


def find_neighbors(edges, v, r, neighbors):
    if r <= 0:
        return
    for n in edges[v]:
        if n in neighbors:
            continue
        neighbors.append(n)
        find_neighbors(edges, n, r - 1, neighbors)


def avg_neighbors(vertices, edges, v, r):
    neighbors = [v]
    find_neighbors(edges, v, r, neighbors)
    s = 0
    ws = 0
    for n in neighbors:
        if n == v:
            continue
        w = 1 / np.linalg.norm(vertices[n] - vertices[v])
        s += w * vertices[n]
        ws += w

    s /= ws
    return s


def smooth(mesh, times=1, level=30):
    """
    taubin平滑
    :param mesh: 需要平滑的网格
    :param times: 平滑迭代的次数, 1~2次可以去噪声，5~10次可以去坑洞，40~50次可以去除布褶，建议不要超过60次
    :param level: 平滑卷积的范围，通常是1不要修改
    :return: 平滑网格
    """
    # for i in range(times):
    #     taubin(mesh, 0.8, 1)
    #     taubin(mesh, -0.8, 1)
    smooth_hcl(mesh, times)
    return mesh


def smooth_laplacian(mesh, times=1, level=1):
    """

    :param mesh: 需要平滑的网格
    :param times: 平滑迭代的次数, 1~2次可以去噪声，5~10次可以去坑洞，40~50次可以去除布褶，建议不要超过60次
    :param level: 平滑卷积的范围，通常是1不要修改
    :return: 平滑网格
    """
    for i in range(times):
        new_vertices = []
        new_vertices.extend(mesh.vertices)
        for v in mesh.edges:
            neighbors = [v]
            find_neighbors(mesh.edges, v, level, neighbors)
            s = 0
            ws = 0
            for n in neighbors:
                bias = 1
                if n in mesh.bounds:
                    bias = 10
                    if n == v:
                        bias = 50
                w = 1 / (np.dot(mesh.vertices[n] - mesh.vertices[v], mesh.vertices[n] - mesh.vertices[v]) + 0.002) * bias
                s += w * mesh.vertices[n]
                ws += w
            if ws > 0:
                new_vertices[v] = s / ws
        mesh.vertices = np.array(new_vertices)
    return mesh


def taubin(mesh, mu, level=1):
    new_vertices = []
    new_vertices.extend(mesh.vertices)
    for v in mesh.edges:
        neighbors = [v]
        find_neighbors(mesh.edges, v, level, neighbors)
        s = 0
        ws = 0
        for n in neighbors:
            diff = mesh.vertices[n] - mesh.vertices[v]
            w = 1
            if n in mesh.bounds:
                w = 10
                if n == v:
                    w = 50
            s += diff * w
            ws += w
        if ws > 0:
            new_vertices[v] = mesh.vertices[v] + mu * s / ws
    mesh.vertices = np.array(new_vertices)


def get_weight_hcl(origin, mesh, n, v):
    w = 1
    if n in mesh.bounds:
        w = 10
        if n == v:
            w = 50
    return w


def push_back_vertex(origin, mesh, mu):
    for v in range(len(mesh.vertices)):
        w = 1 - mu
        mesh.vertices[v] = mesh.vertices[v] * w + origin.vertices[v] * (1 - w)


def smooth_hcl(mesh, times):
    origin = Mesh(mesh)
    for i in range(times):
        taubin(mesh, 1, 1)
        push_back_vertex(origin, mesh, 0.2)
        taubin(mesh, -0.2, 1)
    return mesh


if __name__ == '__main__':
    # m = Mesh().load('../data/beta_simulation/result/1.obj')
    m = Mesh().load('../data/pose_simulation/tst/shape_y_final/5.obj')
    smooth_hcl(m, 50)
    # expand(m, 1.03)
    m.save('../tst/save_mesh.obj')


def smooth_bounds(mesh, times):
    def taubin(mesh, mu, level=1):
        new_vertices = []
        new_vertices.extend(mesh.vertices)
        for v in mesh.bounds:
            neighbors = [v]
            find_neighbors(mesh.edges, v, level, neighbors)
            s = 0
            ws = 0
            for n in neighbors:
                diff = mesh.vertices[n] - mesh.vertices[v]
                w = 0
                if n in mesh.bounds:
                    w = 10
                    if n == v:
                        w = 30
                s += diff * w
                ws += w
            if ws > 0:
                new_vertices[v] = mesh.vertices[v] + mu * s / ws
        mesh.vertices = np.array(new_vertices)

    for i in range(times):
        taubin(mesh, 1)
