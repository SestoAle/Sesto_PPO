# pylint: disable=no-member
""" scatter using MarkersVisual """

import numpy as np
import sys
import threading
from PyQt5 import QtWidgets, QtCore

from matplotlib import cm
from vispy import app, visuals, scene

from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFrame, QStackedLayout)

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
        self.size = (1920, 1024)

    def set_line(self, line):
        self.line = line

    def set_visuals(self, visual):
        self.visuals.append(visual)

    def set_view(self, view):
        self.view = view

    def set_camera(self, camera):
        self.camera = camera

    def on_key_press(self, event):

        print(event.key.name)
        self.map.visible = not self.map.visible

    def rotate(self):
        self.timer = threading.Timer(1/60, self.rotate)
        self.timer.start()
        self.camera.azimuth = self.camera.azimuth + 10

    def rotate_slider(self, value):
        self.camera.azimuth = self.start_azimuth + value*10

    def on_mouse_press(self, event):
        self.print_mouse_event(event, 'Mouse press')

    def print_mouse_event(self, event, what):
        modifiers = ', '.join([key.name for key in event.modifiers])
        print('%s - pos: %r, button: %s, modifiers: %s, delta: %r' %
              (what, event.pos, event.button, modifiers, event.delta))


    def create_new_line(self, line):
        Line3d = scene.visuals.create_visual_node(visuals.LineVisual)
        p2 = Line3d(parent=self.view.scene)
        p2.set_data(line[self.index: self.index + 2])
        print(line[self.index: self.index + 2])
        self.index += 1


class WorldModel(QDialog):
    def __init__(self, canvas, parent=None):
        super(WorldModel, self).__init__(parent)

        self.originalPalette = QApplication.palette()

        styleComboBox = QComboBox()
        styleComboBox.addItems(QStyleFactory.keys())

        styleLabel = QLabel("&Style:")
        styleLabel.setBuddy(styleComboBox)

        self.useStylePaletteCheckBox = QCheckBox("&Use style's standard palette")
        self.useStylePaletteCheckBox.setChecked(True)

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.useStylePaletteCheckBox)

        progressBar = QProgressBar()
        progressBar.setRange(0, 10000)
        progressBar.setValue(0)

        defaultPushButton = QPushButton("Default Push Button")
        defaultPushButton.setDefault(True)
        defaultPushButton.clicked.connect(lambda x: progressBar.setVisible(progressBar.isHidden()) )
        defaultPushButton.resize(50, 50)

        label = QLabel("asjdasjkh")
        label.setStyleSheet('color: black')

        a = QPushButton(canvas.native)
        a.resize(50, 50)
        b = QPushButton("bv")

        stackLayout = QStackedLayout()
        # stackLayout.addWidget(canvas.native)
        # stackLayout.addWidget(label)
        stackLayout.addWidget(canvas.native)

        label = QLabel(canvas.native)
        label.setText("SADASD")
        label.setStyleSheet('color: white')

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(stackLayout)
        mainLayout.addWidget(defaultPushButton)
        mainLayout.addWidget(progressBar)

        self.setLayout(mainLayout)


# run
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        # build your visuals, that's all
        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        # Line3d = scene.visuals.create_visual_node(visuals.LineVisual)

        # The real-things : plot using scene
        # build canvas
        canvas = Canvas(keys='interactive', show=True)

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

        grid = canvas.central_widget.add_grid(margin=10)
        grid.spacing = 0

        title = scene.Label("Plot Title", color='white')
        title.height_max = 40
        grid.add_widget(title, row=0, col=0, col_span=2)

        app = QApplication(sys.argv)
        gallery = WorldModel(canvas)
        gallery.show()
        sys.exit(app.exec_())
