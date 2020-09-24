# coding: utf-8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
import numpy as np
import time as time
from com.mesh.mesh import Mesh


def gen_mesh_array(mesh: Mesh, color):
    verts = mesh.vertices
    norms = mesh.normal
    colors = np.broadcast_to(color, verts.shape)
    return np.hstack((verts, norms, colors)).astype('f')


def gen_mesh_element_array(mesh: Mesh):
    faces = mesh.faces
    return np.array(faces, 'i')


def draw_mesh(mesh, color):
    m_vbo.set_array(gen_mesh_array(mesh, color))
    m_vbo.bind()
    ebo.bind()
    glDrawElements(GL_TRIANGLES, len(mesh.faces * 3), GL_UNSIGNED_INT, ebo)
    # glFlush()


def init_array_renderer():
    glShadeModel(GL_SMOOTH)
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    m_vbo.bind()
    glVertexPointer(3, GL_FLOAT, 36, m_vbo)
    glNormalPointer(GL_FLOAT, 36, m_vbo + 12)
    glColorPointer(3, GL_FLOAT, 36, m_vbo + 24)


class MeshRenderer:
    def __init__(self, mesh, color=(0, 0, 0)):
        self.mesh = mesh
        self.ebo = vbo.VBO(data=gen_mesh_element_array(mesh),target=GL_ELEMENT_ARRAY_BUFFER, usage=GL_STATIC_DRAW)
        self.color = color

    def render(self, color=None):
        if color is not None:
            self.color = color
        m_vbo.set_array(gen_mesh_array(self.mesh, self.color))
        m_vbo.bind()
        self.ebo.bind()
        glDrawElements(GL_TRIANGLES, len(self.mesh.faces) * 3, GL_UNSIGNED_INT, self.ebo)


m_vbo = vbo.VBO(np.array([]))

ebo = vbo.VBO(data=np.array([]),target=GL_ELEMENT_ARRAY_BUFFER, usage=GL_STATIC_DRAW)


def set_ebo(mesh):
    ebo.set_array(gen_mesh_element_array(mesh))


if __name__ == '__main__':
    """
    """

