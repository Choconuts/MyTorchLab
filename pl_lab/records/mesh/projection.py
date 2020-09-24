import math
import numpy as np


def vec3(x=0, y=0, z=0):
    return np.array([x, y, z])


class Plane:
    def __init__(self):
        self.normal = vec3()
        self.point = vec3()

    def from_triangle(self, p1, p2, p3):
        n = np.cross(p2 - p1, p3 - p2)
        self.normal = n / np.linalg.norm(n)
        self.point = p1
        return self


def point_plane_project_vec(v, plane: Plane):
    dot = np.dot((v - plane.point), plane.normal)
    return -dot * plane.normal


def solve_uv(p, p1, p2, p3):
    b = p - p1
    b = np.reshape(b, (3, 1))
    u_factor = p2 - p1
    v_factor = p3 - p1
    w = np.hstack((np.reshape(u_factor, (3, 1)), np.reshape(v_factor, (3, 1))))
    wpinv = np.linalg.pinv(w)
    res = np.matmul(wpinv, b).reshape(2)
    return res[0], res[1]






