# pylint: disable=no-member
""" scatter using MarkersVisual """

import numpy as np
import sys
import threading

from vispy import app, visuals, scene

class Canvas(scene.SceneCanvas):

    def __init__(self, *args, **kwargs):
        self.visuals = []
        self.line = None
        self.index = 0
        self.camera = None
        self.timer = None
        scene.SceneCanvas.__init__(self, *args, **kwargs)
        self.title = 'App demo'

    def set_line(self, line):
        self.line = line

    def set_visuals(self, visual):
        self.visuals.append(visual)

    # def on_close(self, event):
    #     print('closing!')
    #
    # def on_resize(self, event):
    #     print('Resize %r' % (event.size, ))
    #
    def on_key_press(self, event):

        print(event.key.name)

        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        else:
            self.rotate()

    def rotate(self):
        self.timer = threading.Timer(1/60, self.rotate)
        self.timer.start()
        self.camera.azimuth = self.camera.azimuth + 10

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

    def create_new_line(self, line):
        Line3d = scene.visuals.create_visual_node(visuals.LineVisual)
        p2 = Line3d(parent=view.scene)
        p2.set_data(line[self.index: self.index + 2])
        print(line[self.index: self.index + 2])
        self.index += 1
    # def on_draw(self, event):
    #     gloo.clear(color=True, depth=True)

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

canvas.camera = view.camera

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
    return r, g, b, 0.8

colors = []
for i in range(500):
    colors.append(rgb(0,500,i))

# plot ! note the parent parameter
p1 = Scatter3D(parent=view.scene)
p1.set_gl_state('translucent', blend=True, depth_test=True)
p1.set_data(pos, face_color=colors, symbol='o', size=10,
            edge_width=0.5, edge_color=colors)

canvas.set_line(pos)

# run
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()