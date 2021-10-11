# pylint: disable=no-member
""" scatter using MarkersVisual """

import numpy as np
import sys
import threading
from PyQt5 import QtWidgets, QtCore

from matplotlib import cm

print(cm.get_cmap('tab20b')(0))



from vispy import app, visuals, scene
from vispy.visuals.filters import ShadingFilter, WireframeFilter

class Canvas(scene.SceneCanvas):

    def __init__(self, *args, **kwargs):
        self.view = None
        self.visuals = []
        self.line = None
        self.index = 0
        self.camera = None
        self.timer = None
        self.start_azimuth = None
        self.map = None
        scene.SceneCanvas.__init__(self, *args, **kwargs)
        self.title = 'App demo'

    def set_line(self, line):
        self.line = line

    def set_visuals(self, visual):
        self.visuals.append(visual)

    def set_view(self, view):
        self.view = view

    def set_camera(self, camera):
        self.camera = camera
        # self.start_azimuth = self.camera.azimuth

    # def on_close(self, event):
    #     print('closing!')
    #
    # def on_resize(self, event):
    #     print('Resize %r' % (event.size, ))
    #
    def on_key_press(self, event):

        print(event.key.name)
        self.map.visible = not self.map.visible
        # print(self.camera.center)

        # self.forward()

        # if self.timer is not None:
        #     self.timer.cancel()
        #     self.timer = None
        # else:
        #     self.rotate()

    def rotate(self):
        self.timer = threading.Timer(1/60, self.rotate)
        self.timer.start()
        self.camera.azimuth = self.camera.azimuth + 10

    def rotate_slider(self, value):
        self.camera.azimuth = self.start_azimuth + value*10

    #
    # def on_key_release(self, event):
    #     modifiers = [key.name for key in event.modifiers]
    #     print('Key released - text: %r, key: %s, modifiers: %r' % (
    #         event.text, event.key.name, modifiers))
    #
    def on_mouse_press(self, event):
        self.print_mouse_event(event, 'Mouse press')
    #
    # def on_mouse_release(self, event):
    #     self.print_mouse_event(event, 'Mouse release')
    #
    # def on_mouse_move(self, event):
    #     self.print_mouse_event(event, 'Mouse move')
    #
    # def on_mouse_wheel(self, event):
    #     self.print_mouse_event(event, 'Mouse wheel')

    def print_mouse_event(self, event, what):
        modifiers = ', '.join([key.name for key in event.modifiers])
        print('%s - pos: %r, button: %s, modifiers: %s, delta: %r' %
              (what, event.pos, event.button, modifiers, event.delta))

        self.create_new_line(self.line)

    def forward(self):
        up, forward, right = self.camera._get_dim_vectors()
        # Create mapping so correct dim is up
        pp1 = np.array([(0, 0, 0), (0, 0, -1), (1, 0, 0), (0, 1, 0)])
        pp2 = np.array([(0, 0, 0), forward, right, up])
        pos = -self.camera._actual_distance * forward
        print(pos)

    def create_new_line(self, line):
        Line3d = scene.visuals.create_visual_node(visuals.LineVisual)
        p2 = Line3d(parent=self.view.scene)
        #             vertex_colors=[
        #                 [1, 1, 1, 1],
        #                 [1, 1, 1, 1],
        #                 [1, 1, 1, 1],
        #                 [1, 1, 1, 1],
        #             ])
        # p2.shading_filter.enabled=False
        p2.set_data(line[self.index: self.index + 2])
        print(line[self.index: self.index + 2])
        self.index += 1
    # def on_draw(self, event):
    #     gloo.clear(color=True, depth=True)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow, canvas):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(440, 299)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frameFor3d = QtWidgets.QFrame(self.centralwidget)
        self.frameFor3d.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameFor3d.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameFor3d.setObjectName("frameFor3d")
        self.gridLayout.addWidget(self.frameFor3d, 0, 0, 1, 1)
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalSlider.valueChanged.connect(lambda value: canvas.rotate_slider(value))

        self.gridLayout.addWidget(self.horizontalSlider, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 440, 18))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

class myWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(myWindow, self).__init__()
        self.ui = Ui_MainWindow()


        # build your visuals, that's all
        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        # Line3d = scene.visuals.create_visual_node(visuals.LineVisual)

        # The real-things : plot using scene
        # build canvas
        canvas = Canvas(keys='interactive', show=True)
        self.ui.setupUi(self, canvas)

        # Add a ViewBox to let the user zoom/rotate
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.distance = 500


        canvas.set_camera(view.camera)
        canvas.view = view

        # data
        n = 500
        pos = np.zeros((n, 3))
        colors = np.ones((n, 4), dtype=np.float32)
        radius, theta, dtheta = 1.0, 0.0, 10.5 / 180.0 * np.pi
        for i in range(500):
            theta += dtheta
            x = 0.0 + radius * np.cos(theta)
            y = 0.0 + radius * np.sin(theta)
            z = 1.0 * radius
            r = 10.1 - i * 0.02
            radius -= 0.45
            pos[i] = x, y, z
            colors[i] = (i/500, 1.0-i/500, 0, 0.8)

        def rgb(minimum, maximum, value):
            minimum, maximum = float(minimum), float(maximum)
            ratio = 2 * (value-minimum) / (maximum - minimum)
            b = int(max(0, 255*(1 - ratio)))
            r = int(max(0, 255*(ratio - 1)))
            g = 255 - b - r
            r /= 255
            b /= 255
            g /= 255
            return r, g, b, 1

        colors = []
        for i in range(500):
            colors.append(rgb(0,500,i))

        # plot ! note the parent parameter
        p1 = Scatter3D(parent=view.scene)
        p1.set_gl_state('opaque', blend=True, depth_test=True)
        p1.set_data(pos, face_color=colors, symbol='o', size=10,
                    edge_width=0, scaling=True)

        scene.widgets.Label("ASDKHS", rotation=0.0)

        canvas.set_line(pos)
        canvas.map = p1

        lay = QtWidgets.QVBoxLayout(self.ui.frameFor3d)
        lay.addWidget(canvas.native)

# run
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        appa = QtWidgets.QApplication([])
        application = myWindow()
        application.show()

        app.run()
