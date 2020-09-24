import numpy as np
import time
from com.mesh.mesh import *
import math
from com.path_helper import *

def get_closest_points(cloth, body):
    """
    :param cloth: 7366 vertices
    :param body: 6890 vertices
    :return: 每个衣服顶点最近的人体顶点标号的列表
    """
    # hash the body
    class LinearFunc:
        """
        先用二次把
        """
        def __init__(self, thrs, ks): # [?, 0.5, 0.7], [1.2, 1, ?]
            import copy
            self.thrs = copy.deepcopy(thrs)
            self.thrs.insert(0, 0)
            self.ks = copy.deepcopy(ks)
            k = 1
            for i in range(len(ks)):
                k -= ks[i] * (self.thrs[i + 1] - self.thrs[i])
            self.ks.append(k / (1 - self.thrs[len(ks)]))
            self.bs = [0]
            for i in range(len(ks)):
                self.bs.append(self.bs[len(self.bs) - 1] + ks[i] * (self.thrs[i + 1] - self.thrs[i]))

        def fin(self, x):
            y = 0
            for i in range(len(self.thrs) - 1):
                y += (self.ks[i] * (x - self.thrs[i]) + self.bs[i]) * (self.thrs[i] < x < self.thrs[i + 1])
            return x

        def fout(self, x):
            return x ** 2

    lf = LinearFunc([0.2, 0.5], [2, 1.5])
    resolution = 80

    min_x = np.min(body.vertices)
    max_x = np.max(body.vertices)

    def my_hash(x, step):
        x = (x - min_x) / (max_x - min_x + 0.0001)
        y = int(lf.fin(x) / step)
        assert y >= 0
        return y

    # 建立多层哈希
    resolu = resolution
    resolutions = []
    hash_tables = []
    lists_map = []
    while resolu > 0:
        resolutions.append(resolu)
        step = 1 / resolu
        hash_table = np.zeros((resolu, resolu, resolu), np.int)
        lists = [[]]
        i = 1
        vi = 0
        for v in body.vertices:
            v0 = my_hash(v[0], step)
            v1 = my_hash(v[1], step)
            v2 = my_hash(v[2], step)
            if hash_table[v0][v1][v2] == 0:
                hash_table[v0][v1][v2] = i
                lists.append([vi])
                i += 1
            else:
                lists[hash_table[v0][v1][v2]].append(vi)
            vi += 1


        lists_map.append(lists)
        hash_tables.append(hash_table)
        resolu = int(resolu / 5)

    # timer.tick('hash')

    res = []
    for v in cloth.vertices:
        search_list = []
        layer = 0
        while True:
            if layer >= len(resolutions):
                raise Exception('hash grid not found')
            step = 1 / resolutions[layer]
            v0 = my_hash(v[0], step)
            v1 = my_hash(v[1], step)
            v2 = my_hash(v[2], step)
            if hash_tables[layer][v0][v1][v2] == 0:
                layer += 1
                continue
            if layer < len(resolutions) - 1:
                layer += 1
            step = 1 / resolutions[layer]
            v0 = my_hash(v[0], step)
            v1 = my_hash(v[1], step)
            v2 = my_hash(v[2], step)
            search_list.extend(lists_map[layer][hash_tables[layer][v0][v1][v2]])
            break
        closest_v = -1
        closest_dist = 100
        for p in search_list:
            d = np.linalg.norm(v - body.vertices[p])
            if d < closest_dist:
                closest_dist = d
                closest_v = p
        res.append(closest_v)

    # timer.tick('find')

    def travel():
        if len(body.edges)== 0:
            body.edges = Mesh(body).edges
        for i in range(len(cloth.vertices)):
            now = res[i]
            vc = cloth.vertices[i]
            while True:
                min_d = np.linalg.norm(vc - body.vertices[now])
                min_v = -1
                for v in body.edges[res[i]]:
                    d = np.linalg.norm(vc - body.vertices[v])
                    if d < min_d:
                        min_d = d
                        min_v = v
                if min_v < 0:
                    break
                now = min_v
            res[i] = now
    travel()

    return res


class Timer:
    def __init__(self, prt=True):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.print = prt

    def tick(self, msg=''):
        res = time.time() - self.last_time
        if self.print:
            print(msg, res)
        self.last_time = time.time()
        return res

    def tock(self, msg=''):
        if self.print:
            print(msg, time.time() - self.last_time)
        return time.time() - self.last_time


def get_closest_faces(cloth: Mesh, body: Mesh, grid_width = 0.015):

    def gridify(x):
        return np.floor(x / grid_width).astype('i')

    def reduce(func1, func2):
        return func1(func2(cloth.vertices), func2(body.vertices))
    min_coord = reduce(min, np.min)
    bias = 0    # 让所有坐标都大于等于0
    if min_coord < 0:
        bias = -min_coord
    # 不会超过的id值
    power_limit = gridify(reduce(max, np.max) + bias) + 1

    print(power_limit)

    def hash_id(grid_ids):
        hash_code = 0  # 不会超过limit的4次方
        for i in range(len(grid_ids)):
            hash_code += power_limit ** i * grid_ids[i]
        return hash_code

    # 返回一个点周围半径r格以内的所有hash code
    def neighbor_hash(grid_vec, r):
        codes = []
        for x in range(grid_vec[0] - r, grid_vec[0] + r + 1):
            for y in range(grid_vec[1] - r, grid_vec[1] + r + 1):
                for z in range(grid_vec[2] - r, grid_vec[2] + r + 1):
                    codes.append(hash_id([x, y, z]))
        return codes

    body_table = dict()
    for i in range(len(body.vertices)):
        v = body.vertices[i]
        hash_code = hash_id(gridify(v))
        if hash_code in body_table:
            body_table[hash_code].append(i)
        else:
            body_table[hash_code] = [i]

    cloth_vert_num = cloth.vertices.__len__()
    closest_vertices_mapping = np.zeros(cloth_vert_num, 'i')
    closest_faces_mapping = np.zeros(cloth_vert_num, 'i')
    closest_faces_uv = np.zeros((cloth_vert_num, 2))

    search_radius = 2

    def solve_uv(p, p1, p2, p3):
        b = p - p1
        b = np.reshape(b, (3, 1))
        u_factor = p2 - p1
        v_factor = p3 - p1
        w = np.hstack((np.reshape(u_factor, (3, 1)), np.reshape(v_factor, (3, 1))))
        wpinv = np.linalg.pinv(w)
        res = np.matmul(wpinv, b).reshape(2)
        return res[0], res[1]

    bvs = body.vertices
    cvs = cloth.vertices
    for i in range(len(cvs)):

        min_dist = math.inf
        min_i = -1

        close_faces_set = set()

        v = cvs[i]
        ids = gridify(v)
        neighbor_codes = neighbor_hash(ids, search_radius)
        for c in neighbor_codes:
            if c in body_table:
                for bi in body_table[c]:
                    # 把顶点相关的面记录下来
                    for f in body.vertex_face_map[bi]:
                        close_faces_set.add(f)
                    ds = np.dot(bvs[bi] - v, bvs[bi] - v)
                    if ds < min_dist:
                        min_dist = ds
                        min_i = bi

        closest_vertices_mapping[i] = min_i

        def point_to_face_dist(v, a, b, c):
            n = calc_normal(a, b, c)
            n = n / np.linalg.norm(n)
            dot = np.dot((v - a), n)
            return abs(dot), v - dot * n

        # 计算最近的面
        min_face_dist = math.sqrt(min_dist)
        min_face_i = -1
        min_uv = [-1, -1]

        for f in close_faces_set:
            face = body.faces[f]
            ds, anchor = point_to_face_dist(v, bvs[face[0]], bvs[face[1]], bvs[face[2]])
            # 如果没有落在内部
            uu, vv = solve_uv(anchor, bvs[face[0]], bvs[face[1]], bvs[face[2]])
            if not (uu >= 0 and vv >= 0 and uu + vv <= 1):
                continue
            # 如果落在内部
            if ds < min_face_dist:
                min_face_dist = ds
                min_face_i = f
                min_uv = [uu, vv]
        closest_faces_mapping[i] = min_face_i
        closest_faces_uv[i] = min_uv

    return closest_vertices_mapping, closest_faces_mapping, closest_faces_uv


class ClosestVertex:
    def __init__(self):
        self.relation = []
        self.relation0 = []

    def get_rela(self):
        return self.relation

    def save(self, file):
        save_json(self.relation, file)
        return self

    def load(self, file):
        import json
        with open(file, 'r') as fp:
            self.relation = np.array(json.load(fp))
            self.relation0 = np.copy(self.relation)
        return self

    def reset(self):
        self.relation = np.copy(self.relation0)

    def calc(self, cloth, body):
        closest_vertices_mapping, closest_faces_mapping, closest_faces_uv = get_closest_faces(cloth, body, 0.02)
        self.relation = closest_vertices_mapping
        return self

    def update(self, cloth, body):
        for i in range(len(cloth.vertices)):
            now = self.relation[i]
            vc = cloth.vertices[i]
            while True:
                min_d = np.linalg.norm(vc - body.vertices[now])
                min_v = -1
                for v in body.edges[self.relation[i]]:
                    d = np.linalg.norm(vc - body.vertices[v])
                    if d < min_d:
                        min_d = d
                        min_v = v
                if min_v < 0:
                    break
                now = min_v
            self.relation[i] = now
        return self

    def calc_rela_once(self, cloth, body):
        rela = []
        append = list.append
        for i in range(len(cloth.vertices)):
            append(rela, self.relation[i])
            vc = cloth.vertices[i]
            while True:
                min_d = np.linalg.norm(vc - body.vertices[rela[i]])
                min_v = -1
                for v in body.edges[self.relation[i]]:
                    d = np.linalg.norm(vc - body.vertices[v])
                    if d < min_d:
                        min_d = d
                        min_v = v
                if min_v < 0:
                    break
                rela[i] = min_v
        return rela


class VertexMapping:
    def __init__(self):
        self.to_mesh = Mesh()
        self.from_mesh = Mesh()
        self.v_map = np.array([])
        self.f_map = np.array([])
        self.uv = np.array([])

    def calc(self):
        self.v_map, self.f_map, self.uv = get_closest_faces(self.to_mesh, self.from_mesh)
        return self

    def save(self, path):
        from com.path_helper import save_json
        obj = dict()
        obj['v'] = self.v_map
        obj['f'] = self.f_map
        obj['u'] = self.uv
        save_json(obj, path)
        return self

    def load(self, path):
        from com.path_helper import load_json
        obj = load_json(path)
        self.v_map = np.array(obj['v'], 'i')
        self.f_map = np.array(obj['f'], 'i')
        self.uv = np.array(obj['u'])
        return self

    def transfer(self, vert_prop_array: np.ndarray):
        shape = list(vert_prop_array.shape)
        shape[0] = len(self.to_mesh.vertices)
        res = np.zeros(shape)
        for i in range(shape[0]):
            uu = self.uv[i][0]
            vv = self.uv[i][1]
            if uu > 0 and vv > 0 and uu + vv < 1:
                def w(ti):
                    return vert_prop_array[self.from_mesh.faces[self.f_map[i]][ti]]

                res[i] = (1 - uu - vv) * w(0) + uu * w(1) + vv * w(2)
            else:
                res[i] = vert_prop_array[self.v_map[i]]
        return res




if __name__ == '__main__':
    from com.posture.smpl import SMPLModel, apply, dis_apply
    smpl = SMPLModel(r'D:\Educate\CAD-CG\GitProjects\LBAC\db\model\smpl.pkl')
    c = Mesh().load(r'D:\Educate\CAD-CG\GitProjects\LBAC\db\gt\beta\1\template.obj')
    b = Mesh().from_vertices(smpl.verts, smpl.faces)
    b.compute_vertex_normal()

    def save_weights():
        get_map()
        cvn = len(c.vertices)
        weights = np.zeros((cvn, 24))
        for i in range(cvn):
            uu = mapping.uv[i][0]
            vv = mapping.uv[i][1]
            if uu > 0 and vv > 0 and uu + vv < 1:
                def w(ti):
                    return smpl.weights[b.faces[mapping.f_map[i]][ti]]
                weights[i] = (1 - uu - vv) * w(0) + uu * w(1) + vv * w(2)
            else:
                weights[i] = smpl.weights[mapping.v_map[i]]
        from com.path_helper import save_json, conf_path

        save_json(weights, conf_path('weights-tst.json', 'tst'))


    def tst_weights():
        from com.path_helper import load_json, conf_path
        weights = np.array(load_json(conf_path(r'model\relation\cloth_weights_7366.json')))
        print(weights)

        smpl.set_params(pose=np.random.random((24, 3))*0.1)

        c.vertices = apply(smpl, weights, c.vertices)
        c.vertices = dis_apply(smpl, weights, c.vertices)
        # apply(smpl, smpl.weights, b.vertices)
        from lbac.display.stage import view_mesh
        view_mesh(c)

    a = 1
    mapping = VertexMapping()
    mapping.to_mesh = c
    mapping.from_mesh = b

    def save_map():
        from com.path_helper import conf_path
        mapping.calc()
        mapping.save(conf_path('tst-mapping.json', 'tst'))

    def get_map():
        from com.path_helper import conf_path
        mapping.load(conf_path('tst-mapping.json', 'tst'))

    def tst_map():
        get_map()
        cvn = len(c.vertices)
        for i in range(cvn):
            c.vertices[i] = b.vertices[b.faces[mapping.f_map[i]][0]]
            # c.vertices[i] = b.vertices[mapping.v_map[i]]
        from lbac.display.stage import view_mesh
        view_mesh(c)

    a = 1
    # save_weights()
    tst_weights()

    # def solve_uv(p, p1, p2, p3):
    #     b = p - p1
    #     b = np.reshape(b, (3, 1))
    #     u_factor = p2 - p1
    #     v_factor = p3 - p1
    #     w = np.hstack((np.reshape(u_factor, (3, 1)), np.reshape(v_factor, (3, 1))))
    #     wpinv = np.linalg.pinv(w)
    #     res = np.matmul(wpinv, b).reshape(2)
    #     return res[0], res[1]
    #
    # def point_to_face_dist(v, a, b, c):
    #     n = calc_normal(a, b, c)
    #     n = n / np.linalg.norm(n)
    #     dot = np.dot((v - a), n)
    #     return abs(dot), v - dot * n
    #
    # p = np.array([0, 1, 0])
    # p1 = np.array([-1, 0, -1])
    # p2 = np.array([1, 0, 0])
    # p3 = np.array([0, 0, 1])
    # ds, anc = point_to_face_dist(p, p1, p2, p3)
    # print(ds, anc)




