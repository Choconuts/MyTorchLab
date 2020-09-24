from com.mesh.simple_display import *
from com.mesh.array_renderer import *


class Lab:
    def __init__(self):
        self.init = [init_array_renderer]
        self.draw = []
        self.key = []
        self.idle = []
        self.spec = []
        self.pen_color = np.array([0.8, 0.5, 0])

    def graphic(self):

        def call(funcs):
            def wrap():
                for func in funcs:
                    func()
            return wrap

        set_init(call(self.init))
        set_display(call(self.draw))
        set_callbacks(call(self.idle))

        run_glut()

    def color(self, color):
        self.pen_color = color

    def add_point(self, p):
        color = np.copy(self.pen_color)
        def point_draw():
            glColor3f(*color)
            glPointSize(5)
            glBegin(GL_POINTS)
            glVertex3d(p[0], p[1], p[2])
            glEnd()
        self.draw.append(point_draw)

    def add_line(self, s, e):
        color = np.copy(self.pen_color)
        def line_draw():
            glColor3f(*color)
            glBegin(GL_LINES)
            glVertex3d(s[0], s[1], s[2])
            glVertex3d(e[0], e[1], e[2])
            glEnd()
        self.draw.append(line_draw)

    def add_triangle(self, tri):
        color = np.copy(self.pen_color)
        def tri_draw():
            glColor3f(*color)
            glBegin(GL_TRIANGLES)
            for v in tri:
                glVertex3d(v[0], v[1], v[2])
            glEnd()
        self.draw.append(tri_draw)
