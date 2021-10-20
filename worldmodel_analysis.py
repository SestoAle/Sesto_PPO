import matplotlib.pyplot as plt
from math import factorial
import os
import pickle
from math import factorial
from copy import deepcopy
import seaborn as sns
import os

sns.set_theme(style="dark")
from scipy.interpolate import splprep

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
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


from vispy import app, visuals, scene, gloo
from vispy.visuals.filters import ShadingFilter, WireframeFilter

from PyQt5 import QtWidgets
from PyQt5.QtCore import QDateTime, Qt, QTimer, QObject
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QFrame, QStackedLayout)

EPSILON = sys.float_info.epsilon
from PyQt5.QtCore import QThread, pyqtSignal
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

class WorlModelCanvas(QObject, scene.SceneCanvas):
    heatmap_signal = pyqtSignal(bool)
    filtering_mean_signal = pyqtSignal(float)
    cluster_size_signal = pyqtSignal(int)
    in_time_signal = pyqtSignal(int)
    im_rew_signal = pyqtSignal(np.ndarray)

    def __init__(self, *args, **kwargs):
        self.current_line = None
        self.lines = None
        self.line_visuals = []
        self.lines_positions = []
        self.im_rews = []
        self.index = -1
        self.timer = None
        self.agent = None
        self.agent_timer = None
        self.agent_speed = 5
        self.animation_index = None
        self.movement_vector = (0, 0, 0)
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
        self.heatmap_in_time = []
        self.label = None
        self.loading = None
        self.cluster_size = 20
        self.smooth_window = 5
        self.trajectory_visualizer = True

        self.mean_moti_thr = 0.04
        self.sum_moti_thr = 16

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

        if event.key.name == '0':
            self.plot_3D_alpha_map(self.world_model, 0)

        if event.key.name == '1':
            self.plot_3D_alpha_map(self.world_model, 1)

        if event.key.name == '4':
            self.plot_3D_alpha_map(self.world_model, 4)

        if event.key.name == '6':
            self.plot_3D_alpha_map(self.world_model, 6)

        if event.key.name == '8':
            self.plot_3D_alpha_map(self.world_model, 8)

        if event.key.name == '9':
            self.plot_3D_alpha_map(self.world_model, 9)

        if event.key.name == 'O':
            self.plot_3D_alpha_map(self.world_model, 10)

        if event.key.name == 'I':
            self.plot_3D_alpha_map(self.world_model, 11)

        if event.key.name == 'Up' or event.key.name == 'Down':

            if event.key.name == 'Up':
                self.index += 1

            if event.key.name == 'Down':
                self.index -= 1

            self.delete_agent()

            self.index = np.clip(self.index, -1, len(self.line_visuals))

            if self.index == -1 or self.index == len(self.line_visuals):
                self.one_line = False
                self.hide_all_lines()
                self.toggle_lines()
                self.im_rew_signal.emit(np.asarray([]))
                return

            line_index = self.index

            self.hide_all_lines()
            self.line_visuals[line_index].visible = True

            if self.im_rews[line_index] is not None:
                # plt.figure()
                # plt.title("im: {}".format(np.sum(self.im_rews[self.index])))
                plot_data = self.im_rews[self.index]
                plot_data = self.savitzky_golay(plot_data, 21, 3)
                # plot_data = (plot_data - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))
                # plt.plot(range(len(plot_data)), plot_data)
                plot_data = np.asarray(plot_data)
                self.im_rew_signal.emit(plot_data)

            self.one_line=True

        if event.key.name == 'A':
            self.delete_agent()
            self.create_agent(self.trajs[self.index][0, :3])
            self.animation_index = 1
            self.move_agent()

        if event.key.name == 'P':
            plt.close('all')
            if self.index == -1 or self.index == len(self.line_visuals):
                return

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

        self.heatmap.visible = False
        self.covermap.visible = False
        self.heatmap.parent = None
        self.covermap.parent = None
        self.heatmap = None
        self.covermap = None

        for h in self.heatmap_in_time:
            h.visible = False
            h.parent = None

        del self.heatmap_in_time[:]
        self.heatmap_in_time = []

        self.remove_lines()

    def remove_lines(self, only_visuals=False):
        for v in self.line_visuals:
            v.visible = False
            v.parent = None

        del self.line_visuals[:]
        self.line_visuals = []

        if not only_visuals:
            del self.im_rews[:]
            self.im_rews = []

            del self.actions[:]
            self.actions = []

            del self.colors[:]
            self.colors = []

            del self.trajs[:]
            self.trajs = []

            self.index = -1

    def set_maps(self, heatmap, covermap):
        self.heatmap = heatmap
        self.covermap = covermap

    def set_line_visuals(self, visual, positions, im_rews=None, actions=None, color=None):
        self.line_visuals.append(visual)
        self.im_rews.append(im_rews)
        self.actions.append(actions)
        self.colors.append(color)
        self.lines_positions.append(positions)

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
        world_model_in_time = []
        pos_buffer = dict()
        pos_buffer_is_grounded = dict()
        pos_buffer_alpha = dict()
        count = 0
        for traj in list(trajectories.values())[:]:
            count += 1
            for state in traj:

                position = np.asarray(state[:3])
                position[0] = (((position[0] + 1) / 2) * 500)
                position[1] = (((position[1] + 1) / 2) * 500)
                position[2] = (((position[2] + 1) / 2) * 60)
                position = position.astype(int)
                pos_key = ' '.join(map(str, position))
                if pos_key in pos_buffer.keys():
                    pos_buffer[pos_key] += 1
                    if state[3] == 1:
                        pos_buffer_is_grounded[pos_key] = state[3]
                        pos_buffer_alpha[pos_key][int(state[-2] * 10)] += 1
                else:
                    pos_buffer[pos_key] = 1
                    pos_buffer_is_grounded[pos_key] = state[3]
                    alphas = np.zeros(11)
                    alphas[int(state[-2] * 10)] = 1
                    pos_buffer_alpha[pos_key] = alphas

            if count % 100 == 0:
                world_model_t = []
                for k in pos_buffer.keys():
                    k_value = list(map(float, k.split(" ")))
                    if pos_buffer_is_grounded[k] == 1:
                        heat = pos_buffer[k]
                        world_model_t.append(k_value[:3] + [heat])
                world_model_in_time.append(world_model_t)

        world_model = []
        for k in pos_buffer.keys():
            k_value = list(map(float, k.split(" ")))
            if pos_buffer_is_grounded[k] == 1:
                heat = pos_buffer[k]
                alpha_heat = pos_buffer_alpha[k]
                world_model.append(k_value[:3] + [heat] + list(alpha_heat))

        return pos_buffer, pos_buffer_alpha, world_model, world_model_in_time

    def plot_3d_map(self, world_model, color_index=3, percentile=95,
                    colors_gradient=[(150, 0, 0), (255, 255, 0), (255, 255, 255)], set_map=True):
        world_model_array = deepcopy(np.asarray(world_model))
        world_model_array[:, color_index] = np.clip(world_model_array[:, color_index],
                                                    np.percentile(world_model_array[:, 3], 100 - percentile),
                                                    np.percentile(world_model_array[:, 3], percentile))

        min_value = np.min(world_model_array[:, color_index])
        max_value = np.max(world_model_array[:, color_index])
        print(min_value)
        print(max_value)

        colors = []
        for i, c in enumerate(world_model[:, color_index]):
            if c != 0:
                colors.append(self.convert_to_rgb(min_value, max_value, world_model_array[i, color_index],
                                                  colors=colors_gradient))
            else:
                colors.append((0,0,0,0))

        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        colors = np.asarray(colors)
        heatmap = Scatter3D(parent=view.scene)
        heatmap.set_gl_state('additive', blend=True, depth_test=True)
        heatmap.set_data(world_model_array[:, :3], face_color=colors, symbol='o', size=0.7, edge_width=0,
                         edge_color=colors,
                         scaling=True)

        Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
        covermap = Scatter3D(parent=view.scene)
        covermap.set_gl_state('additive', blend=True, depth_test=True)
        covermap.set_data(world_model_array[:, :3], face_color=(0.61, 0, 0, 1), symbol='o', size=0.7, edge_width=0,
                          edge_color=(1, 0, 0, 1), scaling=True)
        covermap.visible = False

        if set_map:
            self.set_maps(heatmap, covermap)
        return world_model_array

    def plot_3D_alpha_map(self, world_model, alpha=0):

        self.heatmap.visible = False
        self.plot_3d_map(world_model, color_index=alpha + 3)


    def plot_3d_map_in_time(self, world_model_in_time):
        return
        min_perc = np.percentile(np.asarray(world_model_in_time[-1])[:, 3], 5)
        max_perc = np.percentile(np.asarray(world_model_in_time[-1])[:, 3], 95)
        for world_model in world_model_in_time:
            world_model = np.asarray(world_model)
            world_model[:, 3] = np.clip(world_model[:, 3], min_perc,
                                        max_perc)

            min_value = np.min(world_model[:, 3])
            max_value = np.max(world_model[:, 3])

            colors = []
            for c in world_model[:, 3]:
                colors.append(self.convert_to_rgb(min_value, max_value, c))

            Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
            colors = np.asarray(colors)
            heatmap = Scatter3D(parent=view.scene)
            heatmap.set_gl_state('additive', blend=True, depth_test=True)
            heatmap.set_data(world_model[:, :3], face_color=colors, symbol='o', size=0.7, edge_width=0, edge_color=colors,
                             scaling=True)

            heatmap.visible = False
            self.heatmap_in_time.append(heatmap)


    def show_heatmap_in_time(self, index):
        self.heatmap.visible = False
        self.covermap.visible = False
        for h in self.heatmap_in_time:
            h.visible = False

        if index == len(self.heatmap_in_time):
            self.heatmap.visible = True
            self.heatmap_signal.emit(True)
        else:
            self.heatmap_in_time[index].visible = True

    def start_loading(self):
        self.loading.visible = True

    def stop_loading(self):
        self.loading.visible = False

    def create_agent(self, starting_position=(255, 255, 0)):
        if self.agent == None:
            Cube = scene.visuals.create_visual_node(visuals.CubeVisual)
            self.agent = Cube(parent=self.view.scene, color='white', size=1)
            # Define a scale and translate transformation :
            self.agent.transform = visuals.transforms.STTransform(translate=starting_position)
        else:
            self.agent.visible = True
            self.movement_vector = (0, 0, 0)
            self.agent.transform = visuals.transforms.STTransform(translate=starting_position)

    def delete_agent(self):
        if self.agent is None:
            return
        if self.agent_timer is not None:
            self.agent_timer.cancel()
            self.agent_timer = None
        self.agent.visible = False

    def rotate(self):
        self.timer = threading.Timer(1/60, self.rotate)
        self.camera.azimuth = self.camera.azimuth + 1
        self.timer.start()

    def move_agent(self):

        self.agent_timer = threading.Timer(1/60, self.move_agent)
        to_move = self.trajs[self.index][self.animation_index, :3]
        current_tr = self.agent.transform.translate
        if np.linalg.norm(to_move - current_tr[:3]) < 0.05 or np.linalg.norm(self.movement_vector) < 0.05:
            self.animation_index += 1
            if self.animation_index == len(self.trajs[self.index]):
                self.agent_timer.cancel()
                self.agent_timer = None
                self.delete_agent()
                return
            to_move = self.trajs[self.index][self.animation_index, :3]
            self.movement_vector = to_move - current_tr[:3]
            self.movement_vector = self.movement_vector / self.agent_speed
        current_tr[0] += self.movement_vector[0]
        current_tr[1] += self.movement_vector[1]
        current_tr[2] += self.movement_vector[2]

        self.agent.transform.translate = current_tr
        self.agent_timer.start()


    def extrapolate_trajectories(self, motivation, trajectories, actions):

        if motivation is not None:

            # Filler the state
            # TODO: I do this because the state that I saved is only the points AND inventory, not the complete state
            # TODO: it is probably better to save everything
            filler = np.zeros((66))
            traj_to_observe = []
            episodes_to_observe = []

            # Goal Area 1
            desired_point_y = 1
            goal_area_x = 447
            goal_area_z = 466
            goal_area_y = 1
            goal_area_height = 20
            goal_area_width = 44

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

            # desired_point_y = 1
            # goal_area_x = 442
            # goal_area_z = 38
            # goal_area_y = 1
            # goal_area_height = 65
            # goal_area_width = 46

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
            for keys, traj in zip(list(trajectories.keys())[:], list(trajectories.values())[:]):
                if traj[0][-1] > 0.5:
                    continue
                # to_observe = False
                # for point in traj:
                #     de_point = np.zeros(3)
                #     de_point[0] = ((np.asarray(point[0]) + 1) / 2) * 500
                #     de_point[1] = ((np.asarray(point[1]) + 1) / 2) * 500
                #     de_point[2] = ((np.asarray(point[2]) + 1) / 2) * 40
                #     if np.abs(de_point[0] - 461) < threshold and \
                #             np.abs(de_point[1] - 102) < threshold and \
                #             np.abs(de_point[2] - 15) < threshold:
                #         to_observe = True
                #         break
                #
                # if to_observe:
                for i, point in enumerate(traj):
                    de_point = np.zeros(3)
                    de_point[0] = ((np.asarray(point[0]) + 1) / 2) * 500
                    de_point[1] = ((np.asarray(point[1]) + 1) / 2) * 500
                    de_point[2] = ((np.asarray(point[2]) + 1) / 2) * 60
                    if goal_area_x < de_point[0] < (goal_area_x + goal_area_width) and \
                            goal_area_z < de_point[1] < (goal_area_z + goal_area_height) and \
                            np.abs(de_point[2] - desired_point_y) < threshold:
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

            traj_to_observe = np.asarray(traj_to_observe)
            print(len(traj_to_observe))

            return traj_to_observe, mean_moti_rews_dict, sum_moti_rews_dict

    def filtering_trajectory(self):

        motivation = self.motivation
        traj_to_observe = self.unfiltered_trajs
        mean_moti_rews_dict = self.mean_moti_rews_dict
        sum_moti_rews_dict = self.sum_moti_rews_dict

        mean_moti_rews = list(mean_moti_rews_dict.values())
        sum_moti_rews = list(sum_moti_rews_dict.values())

        print(" ")
        print("Max mean moti: {}".format(np.max(mean_moti_rews)))
        print("Mean mean moti: {}".format(np.mean(mean_moti_rews)))
        print("Min mean moti: {}".format(np.min(mean_moti_rews)))
        print(" ")
        print("Max sum moti: {}".format(np.max(sum_moti_rews)))
        print("Mean sum moti: {}".format(np.mean(sum_moti_rews)))
        print("Min sum moti: {}".format(np.min(sum_moti_rews)))
        print(" ")

        # Get those trajectories that have an high motivation reward AND a low imitation reward
        mean_moti_rews_dict = {k: v for k, v in
                               sorted(mean_moti_rews_dict.items(), key=lambda item: item[1], reverse=True)}

        # im_heatmap = []
        filler = np.zeros((66))
        # for traj in traj_to_observe:
        #     states_batch = []
        #     actions_batch = []
        #     # key = episodes_to_observe[idx]
        #
        #     # for state, action in zip(traj, actions[key]):
        #     for state in traj:
        #         # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
        #         # TODO: correct form
        #         state = np.asarray(state)
        #         state = np.concatenate([state, filler])
        #         state[-2:] = state[3:5]
        #
        #         # Create the states batch to feed the models
        #         state = dict(global_in=state)
        #         states_batch.append(state)
        #
        #     im_rew = motivation.eval(states_batch)
        #     im_rew = self.savitzky_golay(im_rew, 101, 3)
        #     for state, im in zip(states_batch, im_rew):
        #         position = np.zeros(3)
        #         position[0] = (((state['global_in'][0] + 1) / 2) * 500)
        #         position[1] = (((state['global_in'][1] + 1) / 2) * 500)
        #         position[2] = (((state['global_in'][2] + 1) / 2) * 60)
        #         position = position.astype(int)
        #         im_heatmap.append(list(position) + [im])

        moti_to_observe = []
        print(self.mean_moti_thr)
        for k, v in zip(mean_moti_rews_dict.keys(), mean_moti_rews_dict.values()):
            if v > self.mean_moti_thr and sum_moti_rews_dict[k] > self.sum_moti_thr:
                moti_to_observe.append(k)
        moti_to_observe = np.reshape(moti_to_observe, -1)

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
            # key = episodes_to_observe[idx]

            # for state, action in zip(traj, actions[key]):
            for state in traj:
                # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
                # TODO: correct form
                state = np.asarray(state)
                state = np.concatenate([state, filler])
                state[-2:] = state[3:5]

                # Create the states batch to feed the models
                state = dict(global_in=state)
                states_batch.append(state)

            im_rew = motivation.eval(states_batch)
            # im_rew = savitzky_golay(im_rew, 51, 3)
            # im_rew = (im_rew - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))
            all_normalized_im_rews.append(im_rew)
            all_sum_fitlered_im_rews.append(np.sum(im_rew))

        # if False:
        if len(all_normalized_im_rews) > self.cluster_size:
            cluster_indices = cluster(all_normalized_im_rews, clusters=self.cluster_size)
        else:
            cluster_indices = np.arange(len(all_normalized_im_rews))

        # episodes_to_observe = np.asarray(episodes_to_observe)[idxs_to_observe][cluster_indices]
        all_normalized_im_rews = np.asarray(all_normalized_im_rews)

        # for i, traj, im_rews, key in zip(range(len(cluster_indices)),
        #                                  traj_to_observe[idxs_to_observe][cluster_indices],
        #                                  all_normalized_im_rews[cluster_indices], episodes_to_observe):
        for i, traj, im_rews in zip(range(len(cluster_indices)),
                                    traj_to_observe[idxs_to_observe][cluster_indices],
                                    all_normalized_im_rews[cluster_indices]):

            self.im_rews.append(im_rews)
            self.actions.append(None)
            self.trajs.append(traj)

            # self.print_3d_traj(traj, index=i)
            self.print_3d_traj(traj, index=i)

        # im_heatmap = np.asarray(im_heatmap)
        # im_heatmap[:, 3] = (im_heatmap[:, 3] - np.min(im_heatmap[:, 3]))/(np.max(im_heatmap[:, 3]) - np.min(im_heatmap[:, 3]))
        # self.plot_3d_map(im_heatmap, color_index=3, colors_gradient=[(0, 0, 255), (0, 255, 0)], percentile=100,
        #                  set_map=False)

    def load_precomputed_models(self, model_name, folder='arrays'):
        try:
            with open('{}/{}/{}_buffer.pickle'.format(folder, model_name, model_name), 'rb') as f:
                buffer = pickle.load(f)
            with open('{}/{}/{}_buffer_alpha.pickle'.format(folder, model_name, model_name), 'rb') as f:
                buffer_alpha = pickle.load(f)
            with open('{}/{}/{}_worldmodel.npy'.format(folder, model_name, model_name), 'rb') as f:
                world_model = np.load(f, allow_pickle=True)
            with open('{}/{}/{}_worldmodel_time.npy'.format(folder, model_name, model_name), 'rb') as f:
                world_model_in_time = np.load(f, allow_pickle=True)
            with open('{}/{}/{}_stats.pickle'.format(folder, model_name, model_name), 'rb') as f:
                stats = pickle.load(f)
            return buffer, buffer_alpha, world_model, stats, world_model_in_time

        except Exception as e:
            print(e)
            return None, None, None, None, None

    def save_precomputed_models(self, model_name, buffer, buffer_alpha, world_model, stats, world_model_in_time=None,
                                folder='arrays'):
        with open('{}/{}/{}_buffer.pickle'.format(folder, model_name, model_name), 'wb') as f:
            pickle.dump(buffer, f)
        with open('{}/{}/{}_buffer_alpha.pickle'.format(folder, model_name, model_name), 'wb') as f:
            pickle.dump(buffer_alpha, f)
        with open('{}/{}/{}_worldmodel.npy'.format(folder, model_name, model_name), 'wb') as f:
            np.save(f, world_model)
        with open('{}/{}/{}_worldmodel_time.npy'.format(folder, model_name, model_name), 'wb') as f:
            np.save(f, world_model_in_time)
        with open('{}/{}/{}_stats.pickle'.format(folder, model_name, model_name), 'wb') as f:
            pickle.dump(stats, f)

    def load_unfiltered_trajs(self, model_name, folder='arrays'):
        try:
            with open('{}/{}/{}_unf_trajs.npy'.format(folder, model_name, model_name), 'rb') as f:
                unfiltered_trajs = np.load(f, allow_pickle=True)
            with open('{}/{}/{}_moti.pickle'.format(folder, model_name, model_name), 'rb') as f:
                moti = pickle.load(f)

            return unfiltered_trajs, moti

        except Exception as e:
            print(e)
            return None, None

    def save_unfiltered_trajs(self, model_name, trajs, mean_moti, sum_moti, folder='arrays'):
        with open('{}/{}/{}_unf_trajs.npy'.format(folder, model_name, model_name), 'wb') as f:
            np.save(f, trajs)
        with open('{}/{}/{}_moti.pickle'.format(folder, model_name, model_name), 'wb') as f:
            moti = dict(mean=mean_moti, sum=sum_moti)
            pickle.dump(moti, f)

    def load_data_from_disk(self, model_name):
        trajectories = dict()
        actions = dict()
        for filename in os.listdir("arrays/{}/".format(model_name)):
            if 'trajectories' in filename:
                with open("arrays/{}/{}".format(model_name, filename), 'r') as f:
                    trajectories.update(json.load(f))
            elif 'actions' in filename:
                with open("arrays/{}/{}".format(model_name, filename), 'r') as f:
                    actions.update(json.load(f))
        trajectories = {int(k): v for k, v in trajectories.items()}
        trajectories = collections.OrderedDict(sorted(trajectories.items()))
        actions = {int(k): v for k, v in actions.items()}
        actions = collections.OrderedDict(sorted(actions.items()))
        return trajectories, actions

    def load_data(self, model_name):

        self.remove_maps()

        buffer, buffer_alpha, world_model, stats, world_model_in_time = self.load_precomputed_models(model_name)
        unfiltered_trajs, unfiltered_moti = self.load_unfiltered_trajs(model_name)

        self.unfreeze()
        self.world_model = world_model
        self.buffer_alpha = buffer_alpha
        self.freeze()
        trajectories = None
        actions = None

        if buffer is None or world_model is None:

            trajectories, actions = self.load_data_from_disk(model_name)

            buffer, buffer_alpha, world_model, world_model_in_time = self.trajectories_to_pos_buffer(trajectories)
            world_model = np.asarray(world_model)
            stats = dict(episodes=len(trajectories))

            self.save_precomputed_models(model_name, buffer, buffer_alpha, world_model, stats, world_model_in_time)

            unfiltered_trajs = None
            unfiltered_moti = None

        self.in_time_signal.emit(len(world_model_in_time))
        self.plot_3d_map(world_model)
        self.plot_3d_map_in_time(world_model_in_time)

        if self.trajectory_visualizer:
            motivation = self.load_motivation(model_name)

            if unfiltered_trajs is None:

                if trajectories is None:
                    trajectories, actions = self.load_data_from_disk(model_name)

                unfiltered_trajs, mean_moti_rews_dict, sum_moti_rews_dict = \
                    self.extrapolate_trajectories(motivation, trajectories, actions)

                self.save_unfiltered_trajs(model_name, unfiltered_trajs, mean_moti_rews_dict, sum_moti_rews_dict)

            if unfiltered_moti is not None:
                mean_moti_rews_dict, sum_moti_rews_dict = unfiltered_moti.values()

            self.unfreeze()
            self.motivation = motivation
            self.unfiltered_trajs = unfiltered_trajs
            self.mean_moti_rews_dict = mean_moti_rews_dict
            self.sum_moti_rews_dict = sum_moti_rews_dict
            self.freeze()
            self.filtering_mean_signal.emit(self.mean_moti_thr)
            self.cluster_size_signal.emit(self.cluster_size)
            self.filtering_trajectory()

        self.change_text(model_name, len(list(buffer.keys())), stats['episodes'])
        self.stop_loading()

    def load_motivation(self, model_name):
        graph = tf.compat.v1.Graph()
        motivation = None
        reward_model = None
        try:
            # Load motivation model
            with graph.as_default():
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
        return motivation

    def print_3d_traj(self, traj, index=None, with_denorm=True, max=None, min=None):
        """
        Method that will plot the trajectory
        """
        ep_trajectory = np.asarray(traj)
        smooth_window = self.smooth_window
        color = 'c'

        if with_denorm:
            ep_trajectory[:, 0] = ((np.asarray(ep_trajectory[:, 0]) + 1) / 2) * 500
            ep_trajectory[:, 1] = ((np.asarray(ep_trajectory[:, 1]) + 1) / 2) * 500
            ep_trajectory[:, 2] = ((np.asarray(ep_trajectory[:, 2]) + 1) / 2) * 60

        smoothed_traj = deepcopy(ep_trajectory)

        smoothed_traj[:, 0] = self.savitzky_golay(smoothed_traj[:, 0], smooth_window, 3)
        smoothed_traj[:, 1] = self.savitzky_golay(smoothed_traj[:, 1], smooth_window, 3)
        smoothed_traj[:, 2] = self.savitzky_golay(smoothed_traj[:, 2], smooth_window, 3)

        if index is None:
            color = (0, 0.90, 0.90, 1)
        else:
            color = self.random_color(index)

        # color = convert_to_rgb(1, 20, index, colors=[(0, 255, 255), (255, 0, 255)])
        # color = cm.get_cmap('tab20b')(index % 20)
        color = cm.get_cmap('tab20b')(traj[0][-1] % 20)

        if self.im_rews[index] is not None:
            im_rews = deepcopy(self.im_rews[index])
            im_rews = self.savitzky_golay(im_rews, 21, 5)
            colors = []
            for i in im_rews:
                if i > np.mean(np.clip(im_rews, np.percentile(im_rews, 20), np.max(im_rews))):
                    colors.append((1, 0, 0, 1))
                else:
                    colors.append(color)
        else:
            colors = color

        Tube3D = scene.visuals.create_visual_node(visuals.TubeVisual)
        p1 = Tube3D(parent=view.scene, points=smoothed_traj[:, :3], color=colors, radius=0.5)
        # wire = WireframeFilter(width=0.5)
        # p1.attach(wire)
        p1.shading_filter.enabled = False
        self.line_visuals.append(p1)
        self.colors.append(color)
        return p1

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

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.gcf()
        self.axes = plt.gca()
        plt.axis('off')
        plt.clf()
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    def plot(self, x, y):
        plt.clf()
        plt.plot(x, y)
        # plt.xlim(0, 100)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.draw()

    def clear_plot(self):
        plt.clf()
        self.draw()

class WorldModelApplication(QDialog):
    def __init__(self, canvas, parent=None):
        super(WorldModelApplication, self).__init__(parent)

        self.originalPalette = QApplication.palette()
        self.canvas = canvas

        self.modelNameCombo = QComboBox()

        areas = []
        for file in os.listdir('arrays'):
            d = os.path.join('arrays', file)
            if os.path.isdir(d):
                areas.append(file)
        areas.sort()
        self.modelNameCombo.addItems([""] + areas)
        self.last_model_name = ""

        styleLabel = QLabel("&Model Name:")
        styleLabel.setBuddy(self.modelNameCombo)

        self.modelNameCombo.activated[str].connect(self.name_combo_changed)

        self.heatmapCheck = QCheckBox("&heatmap")
        self.heatmapCheck.setChecked(True)
        self.canvas.heatmap_signal.connect(lambda b: self.set_state_checkbox(b))
        self.heatmapCheck.toggled.connect(canvas.change_map)

        topLayout = QHBoxLayout()
        topLayout.addWidget(styleLabel)
        topLayout.addWidget(self.modelNameCombo)
        topLayout.addStretch(1)
        topLayout.addWidget(self.heatmapCheck)

        midLeftLayout = QGridLayout()

        trajsLayout = QVBoxLayout()
        self.timeLabelText = "&Heatmap in time: {}"
        self.timeLabel = QLabel()
        self.timeSlider = QSlider(Qt.Horizontal)
        self.timeLabel.setBuddy(self.timeSlider)
        self.timeSlider.setMaximum(100)
        self.timeSlider.setValue(0)
        self.timeLabel.setText(self.timeLabelText.format(self.timeSlider.value() * 500))
        self.timeSlider.setMinimumSize(200, 0)

        self.alphaCombo = QComboBox()
        self.alphaLabel = QLabel('&alpha:')
        self.alphaLabel.setBuddy(self.alphaCombo)
        self.alphaCombo.addItems(['all', '0.0', '0.3', '0.5', '0.7', '0.8', '0.9', '1.0'])
        self.alphaCombo.activated[str].connect(self.alpha_name_changed)

        self.smoothSlider = QSlider(Qt.Horizontal)
        self.smoothLabel = QLabel("&Smooth:")
        self.smoothLabel.setBuddy(self.smoothSlider)
        self.smoothSlider.setMaximum(101)
        self.smoothSlider.setMinimum(5)
        self.smoothSlider.setTickInterval(2)
        self.smoothSlider.setSingleStep(2)
        self.smoothSlider.sliderReleased.connect(self.smooth_changed)

        trajsLayout.addWidget(self.timeLabel)
        trajsLayout.addWidget(self.timeSlider)
        trajsLayout.addWidget(self.alphaLabel)
        trajsLayout.addWidget(self.alphaCombo)
        trajsLayout.addWidget(self.smoothLabel)
        trajsLayout.addWidget(self.smoothSlider)
        trajsLayout.addStretch(1)
        trajsLayout.setContentsMargins(20, 20, 20, 20)
        self.timeSlider.valueChanged.connect(self.time_slider_changed)
        self.canvas.in_time_signal.connect(lambda x: {
            self.timeSlider.blockSignals(True),
            self.timeSlider.setMaximum(x),
            self.timeSlider.setValue(x),
            self.timeSlider.blockSignals(False)})

        controlLayout = QVBoxLayout()
        controlLayout.addStretch(1)

        self.filteringMean = QSlider(Qt.Horizontal)
        self.filteringMean.setMinimumSize(200, 0)
        self.filteringMeanLabelText = '&Mean IM: {}'
        self.filteringMeanLabel = QLabel(self.filteringMeanLabelText.format(self.filteringMean.value()))
        self.filteringMeanLabel.setBuddy(self.filteringMean)

        self.clusterSize = QSlider(Qt.Horizontal)
        self.clusterSize.setMinimumSize(200, 0)
        self.clusterSize.setMaximum(30)
        self.clusterSizeText = '&Clusters: {}'
        self.clusterSizeLabel = QLabel(self.clusterSizeText.format(self.clusterSize.value()))
        self.clusterSizeLabel.setBuddy(self.clusterSize)

        self.filteringButton = QPushButton('&Filter')
        self.filteringButton.pressed.connect(self.change_thr_filtering)

        controlLayout.addWidget(self.filteringMeanLabel)
        controlLayout.addWidget(self.filteringMean)
        controlLayout.addWidget(self.clusterSizeLabel)
        controlLayout.addWidget(self.clusterSize)
        controlLayout.addWidget(self.filteringButton)
        controlLayout.setContentsMargins(20, 20, 20, 20)

        self.filteringMean.valueChanged.connect(lambda x: self.filteringMeanLabel.setText(
            self.filteringMeanLabelText.format(
                self.normalize_value(x))))

        self.clusterSize.valueChanged.connect(lambda x: self.clusterSizeLabel.setText(
            self.clusterSizeText.format(x)))

        canvas.filtering_mean_signal.connect(lambda x: {
            self.filteringMean.setValue(self.de_normalize(x))
        })

        canvas.cluster_size_signal.connect(lambda x: self.clusterSize.setValue(x))

        midLeftLayout.addLayout(trajsLayout, 0, 0)
        midLeftLayout.addLayout(controlLayout, 1, 0)

        midLayout = QHBoxLayout()
        midLayout.addWidget(canvas.native)
        midLayout.addLayout(midLeftLayout)

        bottomLayout = QVBoxLayout()
        self.plotWidget = MplCanvas()
        bottomLayout.addWidget(self.plotWidget)
        self.plotWidget.setMinimumSize(0, 80)
        self.plotWidget.setMaximumSize(100000, 80)
        self.canvas.im_rew_signal.connect(self.plot_curiosity)

        # bottomLayout.addStretch(1)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(midLayout)
        mainLayout.addLayout(bottomLayout)

        self.load_thread = None
        self.setLayout(mainLayout)
        self.resize(1980, 1024)

    class MyThread(QThread):
        finished = pyqtSignal()

        def __init__(self, function):
            self.function = function
            super(WorldModelApplication.MyThread, self).__init__()

        def run(self):
            self.function()
            self.finished.emit()

    def set_state_checkbox(self, b):
        self.heatmapCheck.blockSignals(True)
        self.heatmapCheck.setChecked(b)
        self.heatmapCheck.blockSignals(False)

    def smooth_changed(self):
        value = self.smoothSlider.value()
        if value % 2 == 0:
            value -= 1
        self.canvas.smooth_window = value
        self.canvas.remove_lines(only_visuals=True)
        for i, traj in enumerate(self.canvas.trajs):
            p = canvas.print_3d_traj(traj, index=i, with_denorm=False)
            if self.canvas.index != i and self.canvas.index != -1 and self.canvas.index != len(self.canvas.trajs):
                p.visible = False

    def name_combo_changed(self, model_name):
        if model_name == "" or self.last_model_name == model_name:
            return

        self.load_thread = WorldModelApplication.MyThread(function=lambda : self.canvas.load_data(model_name))
        self.last_model_name = model_name
        self.canvas.start_loading()
        self.load_thread.start()
        self.load_thread.finished.connect(self.enable_inputs)
        self.disable_inputs()

    def plot_curiosity(self, x):
        if len(x) == 0:
            self.plotWidget.clear_plot()
        else:
            self.plotWidget.plot(np.arange(len(x)), x)

    def alpha_name_changed(self, value):
        if value=='all':
            value = 0
            self.timeSlider.setEnabled(True)
            self.timeSlider.blockSignals(True)
            self.timeSlider.setValue(self.timeSlider.maximum())
            self.timeSlider.blockSignals(False)
        else:
            self.timeSlider.setEnabled(False)
            value = int(float(value) * 10) + 1
        self.canvas.plot_3D_alpha_map(self.canvas.world_model, value)

    def time_slider_changed(self, value):
        if value != self.timeSlider.maximum():
            self.heatmapCheck.setEnabled(False)
        else:
            self.heatmapCheck.setEnabled(True)
        self.canvas.show_heatmap_in_time(np.clip(value, 0, self.timeSlider.maximum()))
        value = value * 500
        self.timeLabel.setText(self.timeLabelText.format(value))

    def disable_inputs(self):
        self.modelNameCombo.setEnabled(False)
        self.heatmapCheck.setEnabled(False)
        self.clusterSize.setEnabled(False)
        self.filteringMean.setEnabled(False)
        self.filteringButton.setEnabled(False)
        self.timeSlider.setEnabled(False)
        self.alphaCombo.setEnabled(False)

    def enable_inputs(self):
        self.modelNameCombo.setEnabled(True)
        self.heatmapCheck.setEnabled(True)
        self.clusterSize.setEnabled(True)
        self.filteringMean.setEnabled(True)
        self.filteringButton.setEnabled(True)
        self.timeSlider.setEnabled(True)
        self.alphaCombo.setEnabled(True)

    def de_normalize(self, value):
        return int(((value - 0.01) / (0.06 - 0.01)) * (100 - 0) + 0)

    def normalize_value(self, value):
       return round(((value - 0) / (100 - 0)) * (0.06 - 0.01) + 0.01, 3)

    def change_thr_filtering(self):
        # The value of the mean threshold in percentage
        value = self.filteringMean.value()
        value = ((value - 0) / (100 - 0)) * (0.06 - 0.01) + 0.01
        canvas.mean_moti_thr = value
        canvas.cluster_size = self.clusterSize.value()
        canvas.remove_lines()
        canvas.filtering_trajectory()

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
        gallery = WorldModelApplication(canvas)
        gallery.show()
        sys.exit(app.exec_())