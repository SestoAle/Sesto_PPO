"""
Demonstration of Tube
"""

import sys
from vispy import scene
from vispy.geometry.torusknot import TorusKnot

from colorsys import hsv_to_rgb
import numpy as np
from pprint import pprint

class WorlModelCanvas(scene.SceneCanvas):

    def __init__(self, *args, **kwargs):
        self.current_line = None
        self.lines = None
        self.line_visuals = []
        self.im_rews = []
        self.index = -1
        self.timer = None
        self.camera = None
        self.actions = []
        self.colors = []
        self.trajs = []
        self.default_colors = (0, 1, 1, 1)
        self.default_color = False
        self.l1 = None
        scene.SceneCanvas.__init__(self, *args, **kwargs)
        self.title = 'App demo'

    def on_key_press(self, event):
        pprint(vars(l1._meshdata))
        print(l1._meshdata._vertices)
        # l1._meshdata._vertices = np.ones_like(l1._meshdata._vertices)
        print(np.shape(l1._meshdata))
        # l1._meshdata._vertices = np.ones_like(l1._meshdata._vertices)
        # l1._meshdata._vertex_colors = None
        l1.set_data(meshdata=l1._meshdata)
        colors = np.ones((len(l1._meshdata._vertices), 4))
        print(np.shape(colors))
        colors[:, ] = (0, 1, 1, 1)
        print(np.shape(colors))
        # print(colors)
        l1._meshdata.set_vertex_colors(colors)
        # l1.mesh_data_changed()
        print(np.shape(l1._meshdata._vertices))

canvas = WorlModelCanvas(keys='interactive')
view = canvas.central_widget.add_view()

points1 = TorusKnot(5, 3).first_component[:-1]
points1[:, 0] -= 20.
points1[:, 2] -= 15.

points2 = points1.copy()
points2[:, 2] += 30.

points3 = points1.copy()
points3[:, 0] += 41.
points3[:, 2] += 30

points4 = points1.copy()
points4[:, 0] += 41.

points5 = points1.copy()
points5[:, 0] += 20.4
points5[:, 2] += 15

colors = np.linspace(0, 1, len(points1))
colors = np.array([hsv_to_rgb(c, 1, 1) for c in colors])

vertex_colors = np.random.random(8 * len(points1))
vertex_colors = np.array([hsv_to_rgb(c, 1, 1) for c in vertex_colors])

l1 = scene.visuals.Tube(points1,
                        shading='flat',
                        color=colors,  # this is overridden by
                                       # the vertex_colors argument
                        # vertex_colors=vertex_colors,
                        tube_points=8)

l2 = scene.visuals.Tube(points2,
                        color=['red', 'green', 'blue'],
                        shading='smooth',
                        tube_points=8)

l3 = scene.visuals.Tube(points3,
                        color=colors,
                        shading='flat',
                        tube_points=8,
                        closed=True)

l4 = scene.visuals.Tube(points4,
                        color=colors,
                        shading='smooth',
                        tube_points=8,
                        mode='lines')

# generate sine wave radii
radii = np.sin(2 * np.pi * 440 * np.arange(points5.shape[0]) / 44000)
radii = (radii + 1.5) / 2

l5 = scene.visuals.Tube(points5,
                        radius=radii,
                        color='white',
                        shading='smooth',
                        closed=True,
                        tube_points=8)

view.add(l1)
view.add(l2)
view.add(l3)
view.add(l4)
view.add(l5)
view.camera = scene.TurntableCamera()
canvas.l1 = l1
# tube does not expose its limits yet
view.camera.set_range((-20, 20), (-20, 20), (-20, 20))
canvas.show()

if __name__ == '__main__':
    if sys.flags.interactive != 1:
        canvas.app.run()
