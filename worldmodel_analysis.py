import matplotlib.pyplot as plt
from math import factorial
import os
import pickle
from math import factorial

import numpy as np
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import sys
from threading import Thread

from architectures.bug_arch_very_acc_final import *
from motivation.random_network_distillation import RND
from clustering.cluster_im import cluster
from clustering.clustering import cluster_trajectories as cluster_simple
from matplotlib import cm
import collections


from vispy import app, visuals, scene, gloo
from vispy.visuals.filters import ShadingFilter, WireframeFilter

from PyQt5.QtCore import QDateTime, Qt, QTimer, QObject
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFrame, QStackedLayout)

EPSILON = sys.float_info.epsilon
from PyQt5.QtCore import QThread, pyqtSignal
import threading

class WorlModelCanvas(QObject, scene.SceneCanvas):
    heatmap_signal = pyqtSignal(bool)

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
        self.view = None
        self.default_colors = (0, 1, 1, 1)
        self.default_color = False
        self.one_line = False
        self.covermap = None
        self.heatmap = None
        self.label = None
        self.loading = None
        super(WorlModelCanvas, self).__init__()
        self.unfreeze()
        scene.SceneCanvas.__init__(self, *args, **kwargs)
        # QObject.__init__(self, *args, **kwargs)
        self.size = (1920, 1024)
        self.title = 'World Model Analysis'
        self.freeze()

    def set_camera(self, camera):
        self.camera = camera

    def set_view(self, view):
        self.view = view

    def set_label(self, label):
        self.label = label

    def set_loading(self,loading):
        self.loading = loading

    def on_key_press(self, event):
        if event.key.name == 'L':
            self.toggle_lines()

        if event.key.name == 'R':
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            else:
                self.rotate()

        if event.key.name == 'Space':
            self.reset_index()

        if event.key.name == 'F1':
            self.change_line_colors()

        if event.key.name == 'F2':
            self.change_map()

        if event.key.name == 'Up' or event.key.name == 'Down':

            if event.key.name == 'Up':
                self.index += 1

            if event.key.name == 'Down':
                self.index -= 1

            self.index = np.clip(self.index, -1, len(self.line_visuals))

            if self.index == -1 or self.index == len(self.line_visuals):
                self.one_line = False
                self.hide_all_lines()
                self.toggle_lines()
                return

            line_index = self.index

            self.hide_all_lines()

            plt.close('all')
            self.line_visuals[line_index].visible = True
            self.one_line=True

        if event.key.name == 'P':
            plt.close('all')
            if self.index == -1 or self.index == len(self.line_visuals):
                return

            if self.im_rews[self.index] is not None:
                plt.figure()
                plt.title("im: {}".format(np.sum(self.im_rews[self.index])))
                plot_data = self.savitzky_golay(self.im_rews[self.index], 51, 3)
                # plot_data = (plot_data - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))
                plt.plot(range(len(plot_data)), plot_data)

            if self.actions[self.index] is not None:
                plt.figure()
                plt.hist(self.actions[self.index])

                plt.show()

    def change_line_colors(self):
        if self.default_color:
            self.default_color = False
            for i, v in enumerate(self.line_visuals):
                v.set_data(meshdata=v._meshdata)
                colors = np.ones((len(v._meshdata._vertices), 4))
                colors[:, ] = self.colors[i]
                v._meshdata.set_vertex_colors(colors)
        else:
            self.default_color = True
            for i, v in enumerate(self.line_visuals):
                v.set_data(meshdata=v._meshdata)
                colors = np.ones((len(v._meshdata._vertices), 4))
                colors[:, ] = self.default_colors
                v._meshdata.set_vertex_colors(colors)

    def rotate(self):
        self.timer = threading.Timer(1/60, self.rotate)
        self.camera.azimuth = self.camera.azimuth + 1
        self.timer.start()

    def toggle_lines(self):
        if self.index == -1 or self.index == len(self.line_visuals) or not self.one_line:
            if self.line_visuals is not None and len(self.line_visuals) > 0:
                for v in self.line_visuals:
                    v.visible = not v.visible
        else:
            self.line_visuals[self.index].visible = not self.line_visuals[self.index].visible

    def hide_all_lines(self):
        if self.line_visuals is not None and len(self.line_visuals) > 0:
            for v in self.line_visuals:
                v.visible = False

    def reset_index(self):
        if self.index == -1 or self.index == len(self.line_visuals):
            return
        if self.one_line:
            tmp_index = self.index
            self.index = -1
            self.hide_all_lines()
            self.toggle_lines()
            self.one_line = False
            self.index = tmp_index
        else:
            self.hide_all_lines()
            plt.close('all')
            self.line_visuals[self.index].visible = True
            self.one_line = True

    def on_mouse_press(self, event):
        return

    def change_map(self):
        if self.heatmap == None:
            return
        self.heatmap.visible = not self.heatmap.visible
        self.covermap.visible = not self.covermap.visible
        to_emit = True if self.heatmap.visible else False
        self.heatmap_signal.emit(to_emit)

    def set_line(self, line):
        self.current_line = line

    def set_lines(self, lines):
        self.lines = lines

    def remove_maps(self):
        if self.heatmap == None:
            return

        self.heatmap.parent = None
        self.heatmap.parent = None
        self.heatmap = None
        self.covermap = None


    def set_maps(self, heatmap, covermap):
        self.remove_maps()

        self.heatmap = heatmap
        self.covermap = covermap

    def set_line_visuals(self, visual, im_rews=None, actions=None, color=None):
        self.line_visuals.append(visual)
        self.im_rews.append(im_rews)
        self.actions.append(actions)
        self.colors.append(color)

    def random_color(self, value):
        return (np.random.uniform(), np.random.uniform(), np.random.uniform(), 1)

    def change_text(self, filename, points, episodes):

        label = '{}\ncoverage of points: {}\ntotal training episodes: {}'.format(filename, points, episodes)

        self.label.text = label

    def convert_to_rgb(self, minval, maxval, val, colors=[(150, 0, 0), (255, 255, 0), (255, 255, 255)]):

        i_f = float(val - minval) / float(maxval - minval) * (len(colors) - 1)
        i, f = int(i_f // 1), i_f % 1
        if f < EPSILON:
            return colors[i][0] / 255, colors[i][1] / 255, colors[i][2] / 255, 1
        else:
            (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i + 1]
            return int(r1 + f * (r2 - r1)) / 255, int(g1 + f * (g2 - g1)) / 255, int(b1 + f * (b2 - b1)) / 255, 1

    def trajectories_to_pos_buffer(self, trajectories):
        world_model = []
        pos_buffer = dict()
        count = 0
        for traj in list(trajectories.values())[-200:]:
            count += 1
            for state in traj:

                position = np.asarray(state[:5])
                position[0] = (((position[0] + 1) / 2) * 500)
                position[1] = (((position[1] + 1) / 2) * 500)
                position[2] = (((position[2] + 1) / 2) * 60)
                position = position.astype(int)
                pos_key = ' '.join(map(str, position))
                if pos_key in pos_buffer.keys():
                    pos_buffer[pos_key] += 1
                else:
                    pos_buffer[pos_key] = 1

        for k in pos_buffer.keys():
            heat = pos_buffer[k]
            k_value = list(map(float, k.split(" ")))
            if k_value[3] == 1:
                world_model.append(k_value[:3] + [heat])

        return pos_buffer, world_model

    def plot_3d_map(self, trajectories):
        buffer, world_model = self.trajectories_to_pos_buffer(trajectories)
        world_model = np.asarray(world_model)
        world_model[:, 3] = np.clip(world_model[:, 3], np.percentile(world_model[:, 3], 5),
                                    np.percentile(world_model[:, 3], 95))

        min_value = np.min(world_model[:, 3])
        max_value = np.max(world_model[:, 3])

        colors = []
        for c in world_model[:, 3]:
            colors.append(self.convert_to_rgb(min_value, max_value, c))

        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        colors = np.asarray(colors)
        heatmap = Scatter3D(parent=view.scene)
        # p1.events.add(mouse_double_click=scene.events.SceneMouseEvent('OHOH', p1))
        heatmap.set_gl_state('additive', blend=True, depth_test=True)
        heatmap.set_data(world_model[:, :3], face_color=colors, symbol='o', size=0.7, edge_width=0, edge_color=colors,
                         scaling=True)

        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        covermap = Scatter3D(parent=view.scene)
        covermap.set_gl_state('additive', blend=True, depth_test=True)
        covermap.set_data(world_model[:, :3], face_color=(0.61, 0, 0, 1), symbol='o', size=0.7, edge_width=0,
                          edge_color=(1, 0, 0, 1), scaling=True)
        covermap.visible = False

        self.set_maps(heatmap, covermap)
        return buffer, world_model

    def start_loading(self):
        self.loading.visible = True

    def stop_loading(self):
        self.loading.visible = False

    def extrapolate_trajectories(self, model_name, trajectories, actions):
        graph = tf.compat.v1.Graph()
        motivation = None
        reward_model = None
        try:
            # Load motivation model
            with graph.as_default():
                # model_name = "asdasdasd"
                tf.compat.v1.disable_eager_execution()
                motivation_sess = tf.compat.v1.Session(graph=graph)
                motivation = RND(motivation_sess, input_spec=input_spec,
                                 network_spec_predictor=network_spec_rnd_predictor,
                                 network_spec_target=network_spec_rnd_target, obs_normalization=False,
                                 obs_to_state=obs_to_state_rnd, motivation_weight=1)
                init = tf.compat.v1.global_variables_initializer()
                motivation_sess.run(init)
                motivation.load_model(name=model_name, folder='saved')
        except Exception as e:
            reward_model = None
            motivation = None
            print(e)

        if motivation is not None:

            # Filler the state
            # TODO: I do this because the state that I saved is only the points AND inventory, not the complete state
            # TODO: it is probably better to save everything
            filler = np.zeros((66))
            traj_to_observe = []
            episodes_to_observe = []

            # Define the desired points to check
            # I will get all the saved trajectories that touch one of these points at least once
            desired_point_x = 35
            desired_point_z = 500

            # Goal Area 1
            # desired_point_y = 1
            # goal_area_x = 447
            # goal_area_z = 466
            # goal_area_y = 1
            # goal_area_height = 20
            # goal_area_width = 44

            # Goal Area 2
            # desired_point_y = 21
            # goal_area_x = 22
            # goal_area_z = 461
            # goal_area_y = 21
            # goal_area_height = 39
            # goal_area_width = 66

            # desired_point_y = 10
            # goal_area_x = 95
            # goal_area_z = 460
            # goal_area_y = 21
            # goal_area_height = 10
            # goal_area_width = 10

            # desired_point_y = 39
            # goal_area_x = 0
            # goal_area_z = 300
            # goal_area_y = 21
            # goal_area_height = 300
            # goal_area_width = 15

            # Goal Area 3
            # desired_point_y = 28
            # goal_area_x = 35
            # goal_area_z = 18
            # goal_area_y = 28
            # goal_area_height = 44
            # goal_area_width = 44

            # Goal Area 4
            desired_point_y = 1
            goal_area_x = 442
            goal_area_z = 38
            goal_area_y = 1
            goal_area_height = 65
            goal_area_width = 46

            # desired_point_y = 21
            # goal_area_x = 454
            # goal_area_z = 103
            # goal_area_y = 21
            # goal_area_height = 5
            # goal_area_width = 5

            threshold = 4

            # Save the motivation rewards and the imitation rewards
            mean_moti_rews = []
            mean_moti_rews_dict = dict()

            sum_moti_rews = []
            sum_moti_rews_dict = dict()

            sum_il_rews = []
            moti_rews = []
            points = []

            step_moti_rews = []
            step_il_rews = []

            # Get only those trajectories that touch the desired points
            for keys, traj in zip(list(trajectories.keys())[-3000:], list(trajectories.values())[-3000:]):
                for i, point in enumerate(traj):
                    de_point = np.zeros(3)
                    de_point[0] = ((np.asarray(point[0]) + 1) / 2) * 500
                    de_point[1] = ((np.asarray(point[1]) + 1) / 2) * 500
                    de_point[2] = ((np.asarray(point[2]) + 1) / 2) * 60
                    if goal_area_x < de_point[0] < (goal_area_x + goal_area_width) and \
                            goal_area_z < de_point[1] < (goal_area_z + goal_area_height) and \
                            np.abs(de_point[2] - desired_point_y) < threshold and \
                            point[-1] <= 0.5:
                        #         if True:
                        traj_to_observe.append(traj)
                        episodes_to_observe.append(keys)
                        break

            # Get the value of the motivation and imitation models of the extracted trajectories
            for key, traj, idx_traj in zip(episodes_to_observe, traj_to_observe, range(len(traj_to_observe))):
                states_batch = []
                actions_batch = []

                for state, action in zip(traj, actions[key]):
                    # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
                    # TODO: correct form
                    state = np.asarray(state)
                    # state[:3] = 2 * (state[:3]/40) - 1
                    state = np.concatenate([state, filler])
                    state[-2:] = state[3:5]
                    # Create the states batch to feed the models
                    state = dict(global_in=state)
                    states_batch.append(state)
                    actions_batch.append(action)
                    de_point = np.zeros(3)
                    de_point[0] = ((np.asarray(state['global_in'][0]) + 1) / 2) * 500
                    de_point[1] = ((np.asarray(state['global_in'][1]) + 1) / 2) * 500
                    de_point[2] = ((np.asarray(state['global_in'][2]) + 1) / 2) * 60

                    if goal_area_x < de_point[0] < (goal_area_x + goal_area_width) and \
                            goal_area_z < de_point[1] < (goal_area_z + goal_area_height) and \
                            np.abs(de_point[2] - desired_point_y) < threshold:
                        break

                # The actions is one less than the states, so add the last state
                state = traj[-1]
                state = np.concatenate([state, filler])
                state[-2:] = state[3:5]
                state = dict(global_in=state)
                states_batch.append(state)

                # il_rew = reward_model.eval(states_batch[:-1], states_batch, actions_batch)
                il_rew = np.zeros(len(states_batch[:-1]))
                step_il_rews.extend(il_rew)
                il_rew = np.sum(il_rew)
                sum_il_rews.append(il_rew)

                moti_rew = motivation.eval(states_batch)
                moti_rews.append(moti_rew)
                step_moti_rews.extend(moti_rew)
                points.extend([k['global_in'] for k in states_batch])
                mean_moti_rew = np.mean(moti_rew)
                mean_moti_rews.append(mean_moti_rew)
                mean_moti_rews_dict[idx_traj] = mean_moti_rew

                sum_moti_rew = np.sum(moti_rew)
                sum_moti_rews.append(sum_moti_rew)
                sum_moti_rews_dict[idx_traj] = sum_moti_rew

            moti_mean = np.mean(mean_moti_rews)
            il_mean = np.mean(sum_il_rews)
            moti_max = np.max(mean_moti_rews)
            moti_min = np.min(mean_moti_rews)
            il_max = np.max(sum_il_rews)
            il_min = np.min(sum_il_rews)
            epsilon = 0.05
            print(np.max(mean_moti_rews))
            print(np.max(sum_il_rews))
            print(np.median(sum_il_rews))
            print(np.median(moti_mean))
            print(moti_mean)
            print(il_mean)
            print(np.min(sum_il_rews))
            print(np.min(mean_moti_rews))
            print(" ")
            print("Max sum moti: {}".format(np.max(sum_moti_rews)))
            print("Mean sum moti: {}".format(np.mean(sum_moti_rews)))
            print("Min sum moti: {}".format(np.min(sum_moti_rews)))
            print(" ")
            print("Min step moti: {}".format(np.min(step_moti_rews)))
            print("Min step IL: {}".format(np.min(step_il_rews)))
            print("Max step moti: {}".format(np.max(step_moti_rews)))
            print("Max step IL: {}".format(np.max(step_il_rews)))
            print("Mean step moti: {}".format(np.mean(step_moti_rews)))
            print("Mean step IL: {}".format(np.mean(step_il_rews)))
            print(" ")

            # Get those trajectories that have an high motivation reward AND a low imitation reward
            mean_moti_rews_dict = {k: v for k, v in
                                   sorted(mean_moti_rews_dict.items(), key=lambda item: item[1], reverse=True)}
            # moti_to_observe = [k for k in sum_moti_rews_dict.keys()]
            moti_to_observe = []
            for k, v in zip(mean_moti_rews_dict.keys(), mean_moti_rews_dict.values()):
                if v > 0.05 and sum_moti_rews_dict[k] > 16:
                    moti_to_observe.append(k)
            moti_to_observe = np.reshape(moti_to_observe, -1)

            traj_to_observe = np.asarray(traj_to_observe)
            idxs_to_observe = moti_to_observe
            print(moti_to_observe)
            print(idxs_to_observe)

            print("The bugged trajectories are {}".format(len(idxs_to_observe)))

            all_normalized_im_rews = []
            all_sum_fitlered_im_rews = []
            # Plot the trajectories
            for traj, idx in zip(traj_to_observe[idxs_to_observe], idxs_to_observe):

                states_batch = []
                actions_batch = []
                key = episodes_to_observe[idx]

                for state, action in zip(traj, actions[key]):
                    # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
                    # TODO: correct form
                    state = np.asarray(state)
                    # state[:3] = 2 * (state[:3] / 40) - 1
                    state = np.concatenate([state, filler])
                    state[-2:] = state[3:5]

                    # Create the states batch to feed the models
                    state = dict(global_in=state)
                    states_batch.append(state)
                    actions_batch.append(action)

                im_rew = motivation.eval(states_batch)
                # im_rew = savitzky_golay(im_rew, 51, 3)
                # im_rew = (im_rew - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))
                all_normalized_im_rews.append(im_rew)
                all_sum_fitlered_im_rews.append(np.sum(im_rew))

            # if False:
            if len(all_normalized_im_rews) > 20:
                cluster_indices = cluster(all_normalized_im_rews, clusters=20)
            else:
                cluster_indices = np.arange(len(all_normalized_im_rews))

            episodes_to_observe = np.asarray(episodes_to_observe)[idxs_to_observe][cluster_indices]
            all_normalized_im_rews = np.asarray(all_normalized_im_rews)

            for i, traj, im_rews, key in zip(range(len(cluster_indices)),
                                             traj_to_observe[idxs_to_observe][cluster_indices],
                                             all_normalized_im_rews[cluster_indices], episodes_to_observe):
                self.print_3d_traj(traj, im_rews, view, actions[key], i, np.max(all_sum_fitlered_im_rews),
                                     np.min(all_sum_fitlered_im_rews))

    def load_data(self, model_name):

        # x = threading.Thread(target=self.start_loading)
        # x.start()

        trajectories = dict()
        actions = dict()
        for filename in os.listdir("arrays/{}/".format(model_name)):
            if 'trajectories' in filename:
                with open("arrays/{}/{}".format(model_name, filename), 'r') as f:
                    trajectories.update(json.load(f))
            else:
                with open("arrays/{}/{}".format(model_name, filename), 'r') as f:
                    actions.update(json.load(f))
        trajectories = {int(k): v for k, v in trajectories.items()}
        trajectories = collections.OrderedDict(sorted(trajectories.items()))
        actions = {int(k): v for k, v in actions.items()}
        actions = collections.OrderedDict(sorted(actions.items()))

        buffer, world_model = self.plot_3d_map(trajectories)

        self.change_text(model_name, len(list(buffer.keys())), len(trajectories))

        self.extrapolate_trajectories(model_name, trajectories, actions)

        self.stop_loading()
        #
        # x = threading.Thread(target=self.stop_loading)
        # x.start()

    def print_3d_traj(self, traj, im_rews, view, actions, index=None, max=None, min=None):
        """
        Method that will plot the trajectory
        """
        ep_trajectory = np.asarray(traj)
        color = 'c'

        ep_trajectory[:, 0] = ((np.asarray(ep_trajectory[:, 0]) + 1) / 2) * 500
        ep_trajectory[:, 1] = ((np.asarray(ep_trajectory[:, 1]) + 1) / 2) * 500
        ep_trajectory[:, 2] = ((np.asarray(ep_trajectory[:, 2]) + 1) / 2) * 60

        if index is None:
            color = (0, 0.90, 0.90, 1)
        else:
            color = self.random_color(index)

        # color = convert_to_rgb(1, 20, index, colors=[(0, 255, 255), (255, 0, 255)])
        color = cm.get_cmap('tab20b')(index % 20)

        Tube3D = scene.visuals.create_visual_node(visuals.TubeVisual)
        p1 = Tube3D(parent=view.scene, points=ep_trajectory[:, :3], color=color, radius=0.5)
        # wire = WireframeFilter(width=0.5)
        # p1.attach(wire)
        p1.shading_filter.enabled = False
        self.line_visuals.append(p1)
        self.im_rews.append(im_rews)
        self.actions.append(actions)
        self.colors.append(color)
        self.trajs.append(ep_trajectory[:, :3])

    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError as msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order + 1)
        half_window = (window_size - 1) // 2
        # precompute coefficients
        b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
        m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
        lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve(m[::-1], y, mode='valid')

class WorldModel(QDialog):
    def __init__(self, canvas, parent=None):
        super(WorldModel, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        self.canvas = canvas

        self.styleComboBox = QComboBox()

        areas = []
        for file in os.listdir('arrays'):
            d = os.path.join('arrays', file)
            if os.path.isdir(d):
                areas.append(file)

        self.styleComboBox.addItems([""] + areas)
        self.last_model_name = ""

        styleLabel = QLabel("&Model Name:")
        styleLabel.setBuddy(self.styleComboBox)

        self.styleComboBox.activated[str].connect(self.combo_changed)

        self.useStylePaletteCheckBox = QCheckBox("&heatmap")
        self.useStylePaletteCheckBox.setChecked(True)
        self.canvas.heatmap_signal.connect(lambda b: self.set_state_checkbox(b))
        self.useStylePaletteCheckBox.toggled.connect(canvas.change_map)

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(self.styleComboBox)
        topLayout.addStretch(1)
        topLayout.addWidget(self.useStylePaletteCheckBox)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(topLayout)
        mainLayout.addWidget(canvas.native)

        self.load_thread = None
        self.setLayout(mainLayout)

    class MyThread(QThread):
        finished = pyqtSignal()

        def __init__(self, function):
            self.function = function
            super(WorldModel.MyThread, self).__init__()

        def run(self):
            self.function()
            self.finished.emit()

    def set_state_checkbox(self, b):
        self.useStylePaletteCheckBox.blockSignals(True)
        self.useStylePaletteCheckBox.setChecked(b)
        self.useStylePaletteCheckBox.blockSignals(False)

    def combo_changed(self, model_name):
        if model_name == "" or self.last_model_name == model_name:
            return

        self.load_thread = WorldModel.MyThread(function=lambda : self.canvas.load_data(model_name))
        self.last_model_name = model_name
        self.canvas.start_loading()
        self.load_thread.start()
        self.load_thread.finished.connect(self.enable_inputs)
        self.disable_inputs()

    def disable_inputs(self):
        self.styleComboBox.setEnabled(False)
        self.useStylePaletteCheckBox.setEnabled(False)

    def enable_inputs(self):
        self.styleComboBox.setEnabled(True)
        self.useStylePaletteCheckBox.setEnabled(True)

if __name__ == '__main__':
    if sys.flags.interactive != 1:

        # build canvas
        canvas = WorlModelCanvas(keys='interactive', show=True)
        view = canvas.central_widget.add_view()
        view.camera = 'turntable'
        view.camera.fov = 45
        view.camera.distance = 500
        view.camera.translate_speed = 100
        view.camera.center = (255, 255, 60)

        canvas.set_camera(view.camera)
        canvas.set_view(view)

        label_grid = canvas.central_widget.add_grid(margin=0)
        loading_grid = canvas.central_widget.add_grid(margin=0)
        label_grid.spacing = 0

        label = scene.Label("", color='white', anchor_x='left',
                            anchor_y='bottom', font_size=8)
        label.width_max = 20
        label.height_max = 20
        label_grid.add_widget(label, row=0, col=0)
        canvas.set_label(label)

        loading_label = scene.Label("Loading...", color='white', font_size=8)
        loading_label.visible = False
        loading_grid.add_widget(loading_label, row=0, col=0)

        canvas.set_loading(loading_label)

        # Build application and pass it the canvas just created
        app = QApplication(sys.argv)
        gallery = WorldModel(canvas)
        gallery.show()
        sys.exit(app.exec_())