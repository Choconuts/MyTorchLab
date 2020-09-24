# coding: utf-8
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo
import numpy as np
import time as time


class View:
    def __init__(self):
        self.lat = 0
        self.lon = 0
        self.scale = 0.5
        self.offset = [0, 0, 0]


views = [View()]


def apply_view(view, rotate=True):
    glTranslatef(view.offset[0], view.offset[1], 0)
    glScalef(view.scale, view.scale, view.scale)
    if rotate:
        glRotatef(view.lat, 1, 0, 0)
        glRotatef(view.lon + 180, 0, 1, 0)


class MouseState:
    def __init__(self):
        self.down = False
        self.x = 0
        self.y = 0
        self.func = 'ROTATE'


mouse_state = MouseState()


def zoom(in_flag):
    if in_flag:
        views[0].scale *= 1.2
    else:
        views[0].scale /= 1.2
    glutPostRedisplay()


def mouse(button, state, x, y):
    mouse_state.down = (state == GLUT_DOWN)
    mouse_state.x = x
    mouse_state.y = y
    if button == 3 or button == 4:
        mouse_state.func = 'SCALE'
        if state == GLUT_UP:
            return
        if button == 3:
            views[0].scale *=1.1
        else:
            views[0].scale /= 1.1
        glutPostRedisplay()
    elif button == GLUT_LEFT_BUTTON:
        mouse_state.func = 'ROTATE'
    elif button == GLUT_RIGHT_BUTTON:
        mouse_state.func = 'TRANSLATE'


def aspect_ratio():
    return glutGet(GLUT_WINDOW_WIDTH) / glutGet(GLUT_WINDOW_HEIGHT)


def directional_light(i, dir, dif):
    dif4 = np.array([dif[0], dif[1], dif[2], 1])
    pos4 = np.array([dir[0], dir[1], dir[2], 0])
    glEnable(GL_LIGHT0 + i)
    glLightfv(GL_LIGHT0 + i, GL_DIFFUSE, dif4)
    glLightfv(GL_LIGHT0 + i, GL_POSITION, pos4)


def ambient_light(a):
    a4 = np.array([a[0], a[1], a[2], 1])
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, a4)


def draw_axis():
    for i in range(3):
        x = [0, 0, 0]
        x[i] = 1

        glColor3f(*x)
        glBegin(GL_LINES)
        glVertex3f(0, 0, 0)
        glVertex3f(*x)
        glEnd()



def draw_func():
    pass


def init_func():
    pass


def display():
    glClearColor(1, 1, 1, 1)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glEnable(GL_POLYGON_OFFSET_FILL)
    glPolygonOffset(1, 1)
    glEnable(GL_COLOR_MATERIAL)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, 1)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, aspect_ratio(), 0.1, 10)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glTranslatef(0, 0, -1)
    glEnable(GL_LIGHTING)
    glEnable(GL_NORMALIZE)
    directional_light(0, [0, 0, 1], [0.6, 0.6, 0.6])
    directional_light(1, [0, 0, -1], [0.6, 0.6, 0.6])
    directional_light(2, [1, 0, 0], [0.6, 0.6, 0.6])
    directional_light(3, [-1, 0, 0], [0.6, 0.6, 0.6])
    ambient_light([0.3, 0.3, 0.3])
    apply_view(views[0])

    draw_func()

    glutSwapBuffers()


def reshape(w, h):
    return


def motion(x, y):
    if not mouse_state.down:
        return
    view = views[0]
    if mouse_state.func == 'ROTATE':
        speed = 0.25
        view.lon += (x - mouse_state.x) * speed
        view.lat += (y - mouse_state.y) * speed
        view.lat = max(-90, min(90, view.lat))
    elif mouse_state.func == 'TRANSLATE':
        speed = 1e-3
        view.offset[0] += (x - mouse_state.x) * speed
        view.offset[1] -= (y - mouse_state.y) * speed
    mouse_state.x = x
    mouse_state.y = y
    glutPostRedisplay()


# interfaces


window_name = "x"


def run_glut():
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE)
    glutInitWindowSize(1280, 720)
    window = glutCreateWindow(bytes(window_name, encoding="utf8"))
    glutReshapeFunc(reshape)
    glutIdleFunc(callbacks.idle)
    glutDisplayFunc(display)
    glutKeyboardFunc(callbacks.keyboard)
    glutSpecialFunc(callbacks.special)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)

    init_func()

    glutMainLoop()


class CallBacks:
    def __init__(self):
        self.idle = None
        self.keyboard = None
        self.special = None


callbacks = CallBacks()


class FPS:
    def __init__(self):
        self.last = time.time()
        self.count = 0

    def idle(self):
        self.count += 1
        if time.time() - self.last >= 1:
            self.last = time.time()
            print('fps: ', self.count)
            self.count = 0


def set_display(draw):
    global draw_func
    draw_func = draw


def set_init(init):
    global init_func
    init_func = init


def set_callbacks(idle=None, key=None, spec=None):
    if idle is not None:
        callbacks.idle = idle
    if key is not None:
        callbacks.keyboard = key
    if spec is not None:
        callbacks.special = spec


if __name__ == '__main__':
    """
    """

