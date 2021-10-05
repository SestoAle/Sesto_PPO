import matplotlib.pyplot as plt
from math import factorial
import os
import pickle
from math import factorial
from scipy.spatial import distance_matrix

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from architectures.bug_arch_very_acc_final import *
from motivation.random_network_distillation import RND
from reward_model.reward_model import GAIL
from clustering.cluster_im import cluster
from clustering.rdp import rdp_with_index
import threading

from vispy import app, visuals, scene, gloo

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

name_good = 'bug_detector_gail_schifo_acc_com_irl_im_3_no_key_5_2_pl_c2=0.1_replay_random_buffer'

model_name = 'play_3_500_2'
reward_model_name = "vaffanculo_im_9000"

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
        scene.SceneCanvas.__init__(self, *args, **kwargs)
        self.title = 'App demo'

    def on_key_press(self, event):
        if event.key.name == 'L':
            self.toggle_lines()

        if event.key.name == 'R':
            if self.timer is not None:
                self.timer.cancel()
                self.timer = None
            else:
                self.rotate()

        try:
            if event.key.name == 'Up' or event.key.name == 'Down':

                if event.key.name == 'Up':
                    self.index += 1

                if event.key.name == 'Down':
                    self.index -= 1

                self.index = np.clip(self.index, -1, len(self.line_visuals))

                if self.index == -1 or self.index == len(self.line_visuals):
                    self.hide_all_lines()
                    self.toggle_lines()
                    return

                line_index = self.index

                self.hide_all_lines()

                self.line_visuals[line_index].visible = True

                plt.close('all')

                if self.im_rews[line_index] is not None:
                    plt.figure()
                    plt.title("im: {}".format(np.sum(self.im_rews[line_index])))
                    plot_data = savitzky_golay(self.im_rews[line_index], 51, 3)
                    plot_data = (plot_data - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))
                    plt.plot(range(len(plot_data)), plot_data)

                if self.actions[line_index] is not None:
                    plt.figure()
                    plt.hist(self.actions[line_index])

                plt.show()



        except Exception as e:
            pass

    def rotate(self):
        self.timer = threading.Timer(1/60, self.rotate)
        self.camera.azimuth = self.camera.azimuth + 1
        self.timer.start()

    def toggle_lines(self):
        if self.index == -1 or self.index == len(self.line_visuals):
            if self.line_visuals is not None and len(self.line_visuals) > 0:
                for v in self.line_visuals:
                    v.visible = not v.visible
        else:
            self.line_visuals[self.index].visible = not self.line_visuals[self.index].visible

    def hide_all_lines(self):
        if self.line_visuals is not None and len(self.line_visuals) > 0:
            for v in self.line_visuals:
                v.visible = False

    def on_mouse_press(self, event):
        return

    def set_line(self, line):
        self.current_line = line

    def set_lines(self, lines):
        self.lines = lines

    def set_line_visuals(self, visual, im_rews=None, actions=None):
        self.line_visuals.append(visual)
        self.im_rews.append(im_rews)
        self.actions.append(actions)

def plot_map(map):
    """
    Method that will plot the heatmap
    """
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(map)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(map.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(map.shape[0] + 1) - .5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

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

import sys
EPSILON = sys.float_info.epsilon
def convert_to_rgb(minval, maxval, val, colors=[(150,0,0), (255,255,0), (255,255,255)]):
    # "colors" is a series of RGB colors delineating a series of
    # adjacent linear color gradients between each pair.
    # Determine where the given value falls proportionality within
    # the range from minval->maxval and scale that fractional value
    # by the total number in the "colors" pallette.
    i_f = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    # Determine the lower index of the pair of color indices this
    # value corresponds and its fractional distance between the lower
    # and the upper colors.
    i, f = int(i_f // 1), i_f % 1  # Split into whole & fractional parts.
    # Does it fall exactly on one of the color points?
    if f < EPSILON:
        return colors[i][0] / 255, colors[i][1] / 255, colors[i][2] / 255, 1
    else:  # Otherwise return a color within the range between them.
        (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i+1]
        return int(r1 + f*(r2-r1))/255, int(g1 + f*(g2-g1))/255, int(b1 + f*(b2-b1))/255, 1

def plot_3d_map(map):
    # build your visuals, that's all
    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
    # The real-things : plot using scene
    # build canvas
    canvas = WorlModelCanvas(keys='interactive', show=True)

    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'turntable'
    view.camera.fov = 45
    view.camera.distance = 500
    view.camera.translate_speed = 100
    view.camera.center = (255, 255, 60)
    canvas.camera = view.camera

    min_value = np.min(map[:,3])
    max_value = np.max(map[:,3])

    colors = []
    for c in map[:,3]:
        colors.append(convert_to_rgb(min_value,max_value, c))

    colors = np.asarray(colors)
    p1 = Scatter3D(parent=view.scene)
    # p1.events.add(mouse_double_click=scene.events.SceneMouseEvent('OHOH', p1))
    p1.set_gl_state('additive', blend=True, depth_test=True)
    p1.set_data(map[:, :3], face_color=colors, symbol='o', size=1,
                edge_width=0.5, edge_color=colors)

    return view, canvas

def print_event(event):
    print('OOOHHHH')

# def plot_3d_map(map):
#     """
#     Method that will plot the heatmap
#     """
#     ax = plt.gca(projection='3d')
#     ax.set_ylim(0, 500)
#     ax.set_xlim(0, 500)
#     ax.set_zlim(0, 60)
#     # Plot the 3D heatmap
#     ax.scatter(map[:,0], map[:,1], map[:,2], s=0.1)

# Insert to the table. Position must be a 2 element vector
# Return the counter of that position
def insert_to_pos_table(pos_buffer, position, tau):

    # Check if the position is already in the buffer
    for k in pos_buffer.keys():
        # If position - k < tau, then the position is already in the buffer
        # Add its counter to one and return it

        # The position are already normalized by the environment
        k_value = list(map(float, k.split(" ")))
        k_value = np.asarray(k_value)
        position = np.asarray(position)

        distance = np.linalg.norm(k_value - position)
        if distance < tau:
            pos_buffer[k] += 1
            return pos_buffer[k]

    pos_key = ' '.join(map(str, position))
    pos_buffer[pos_key] = 1
    return pos_buffer[pos_key]

# Compute the intrinsic reward based on the counter
def compute_intrinsic_reward(counter, max_counter=500, r_max=0.5):
    return r_max * (1 - (counter / max_counter))

def load_demonstrations(dems_name):
    with open('reward_model/dems/' + dems_name, 'rb') as f:
        expert_traj = pickle.load(f)

    return expert_traj

def save_demonstrations(demonstrations, validations=None, name='dems_acc.pkl'):
    with open('reward_model/dems/' + name, 'wb') as f:
        pickle.dump(demonstrations, f, pickle.HIGHEST_PROTOCOL)
    if validations is not None:
        with open('reward_model/dems/vals_' + name, 'wb') as f:
            pickle.dump(validations, f, pickle.HIGHEST_PROTOCOL)

# Since saving the pos buffer is very expensive, but the trajectories are mandatory,
# let's not save the pos_buffer but extract this from trajectories
import collections
def trajectories_to_pos_buffer(trajectories, world_model, tau=1/40):
    pos_buffer = dict()
    count = 0
    for traj in list(trajectories.values())[:30000]:
        count += 1
        # if traj[-1][-1] < 0.4 or traj[-1][-1] > 0.6:
        #     continue
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

            # if len(state) > 5:
            #     if state[4] == 1:
            #         world_model.append(position)

    for k in pos_buffer.keys():
        heat = pos_buffer[k]
        k_value = list(map(float, k.split(" ")))
        #     if state[4] == 1:
        #         world_model.append(position)
        if k_value[3] == 1:
            world_model.append(k_value[:3] + [heat])

    print("Number of points covered by the agent: {}".format(len(list(pos_buffer.keys()))))
    return pos_buffer


def saved_trajectories_to_demonstrations(trajectories, actions, demonstrations):
    '''
    This method will take some trajectories saved from Intrinsic Motivation + Imitation Learning training
    and transform it into a demonstrations that can be used with Imitation Learning.
    TODO: This method is valid only with the current world_model_3d.py script
    '''

    filler = np.zeros(66)
    for traj, acts in zip(trajectories, actions):
        for idx in range(len(traj) - 1):
            # Transform the state into the correct form
            state = traj[idx]
            state = np.asarray(state)
            state = np.concatenate([state, filler])
            state[-2:] = state[3:5]
            # Create the states batch to feed the models
            state = dict(global_in=state)

            # Do the same thing for obs_n
            state_n = traj[idx + 1]
            state_n = np.asarray(state_n)
            state_n = np.concatenate([state_n, filler])
            state_n[-2:] = state_n[3:5]
            # Create the states batch to feed the models
            state_n = dict(global_in=state_n)

            # Get the corresponfing action
            action = acts[idx]

            demonstrations['obs'].extend([state])
            demonstrations['obs_n'].extend([state_n])
            demonstrations['acts'].extend([action])

    return demonstrations

def print_3d_agg_traj(traj, cells, view):
    agg_traj = []

    ep_trajectory = np.asarray(traj)
    ep_trajectory[:, 0] = ((np.asarray(ep_trajectory[:, 0]) + 1) / 2) * 500
    ep_trajectory[:, 1] = ((np.asarray(ep_trajectory[:, 1]) + 1) / 2) * 500
    ep_trajectory[:, 2] = ((np.asarray(ep_trajectory[:, 2]) + 1) / 2) * 60

    for i in range(0, len(traj) - 1):
        p_t = traj[i][:3]
        p_t1 = traj[i + 1][:3]

        p_t = np.reshape(p_t, (1, 3))
        p_t1 = np.reshape(p_t1, (1, 3))

        cell_t = cells[np.argmin(distance_matrix(p_t, cells))]
        cell_t1 = cells[np.argmin(distance_matrix(p_t1, cells))]

        if cell_t != cell_t1:
            if len(agg_traj) == 0:
                agg_traj.append(cell_t)
            agg_traj.append(cell_t1)

    agg_traj = np.asarray(agg_traj)
    Line3D = scene.visuals.create_visual_node(visuals.LineVisual)
    p1 = Line3D(parent=view.scene)
    p1.set_gl_state('opaque', blend=True, depth_test=True)
    p1.set_data(agg_traj[:, :3], width=0.1, color=(0, 0.90, 0.90, 1))

def print_3d_traj(traj, im_rews, view, canvas, actions):
    """
    Method that will plot the trajectory
    """
    ep_trajectory = np.asarray(traj)
    color = 'c'

    if (ep_trajectory[-1, 3:5] == [0.5, 0.5]).all():
        color = 'g'
    elif (ep_trajectory[-1, 3:5] == [0, 1]).all():
        color = 'b'
    elif (ep_trajectory[-1, 3:5] == [1, 0]).all():
        color = 'y'
    elif (ep_trajectory[-1, 3:5] == [0.3, 0.7]).all():
        color = 'm'
    elif (ep_trajectory[-1, 3:5] == [0.7, 0.3]).all():
        color = 'k'

    ep_trajectory[:, 0] = ((np.asarray(ep_trajectory[:, 0]) + 1) / 2) * 500
    ep_trajectory[:, 1] = ((np.asarray(ep_trajectory[:, 1]) + 1) / 2) * 500
    ep_trajectory[:, 2] = ((np.asarray(ep_trajectory[:, 2]) + 1) / 2) * 60

    # ep_trajectory,_ = rdp_with_index(ep_trajectory[:, :3], range(len(ep_trajectory)), 100)
    # ep_trajectory = np.asarray(ep_trajectory)

    # Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
    # p1 = Scatter3D(parent=view.scene)
    # p1.set_gl_state('additive', blend=True, depth_test=True)
    # p1.set_data(ep_trajectory[:, :3], face_color='blue', symbol='o', size=2,
    #             edge_width=0.5, edge_color='blue')
    Line3D = scene.visuals.create_visual_node(visuals.LineVisual)
    p1 = Line3D(parent=view.scene)
    p1.set_gl_state('opaque', blend=True, depth_test=True)
    p1.set_data(ep_trajectory[:, :3], width=0.1, color=(0, 0.90, 0.90, 1))
    canvas.set_line_visuals(p1, im_rews, actions)


def print_traj(traj):
    """
    Method that will plot the trajectory
    """
    ep_trajectory = np.asarray(traj)
    plt.xlim(0, 500)
    plt.ylim(0, 500)
    color = 'c'

    if (ep_trajectory[-1, 3:5] == [0.5, 0.5]).all():
        color = 'g'
    elif (ep_trajectory[-1, 3:5] == [0, 1]).all():
        color = 'b'
    elif (ep_trajectory[-1, 3:5] == [1, 0]).all():
        color = 'y'
    elif (ep_trajectory[-1, 3:5] == [0.3, 0.7]).all():
        color = 'm'
    elif (ep_trajectory[-1, 3:5] == [0.7, 0.3]).all():
        color = 'k'

    # print(ep_trajectory[-1, 3:5])

    ep_trajectory[:, 0] = ((np.asarray(ep_trajectory[:, 0]) + 1) / 2) * 500
    ep_trajectory[:, 1] = ((np.asarray(ep_trajectory[:, 1]) + 1) / 2) * 500
    plt.plot(ep_trajectory[:, 0], ep_trajectory[:, 1], color)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
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

def print_traj_with_diff(traj, diff, thr=None):
    """
    Method that will plot the trajectory
    """
    ep_trajectory = np.asarray(traj)
    plt.xlim(0, 500)
    plt.ylim(0, 500)
    color = 'g'

    if (ep_trajectory[-1, 3:5] == [0.5, 0.5]).all():
        color = 'g'
    elif (ep_trajectory[-1, 3:5] == [0, 1]).all():
        color = 'b'
    elif (ep_trajectory[-1, 3:5] == [1, 0]).all():
        color = 'y'
    elif (ep_trajectory[-1, 3:5] == [0.3, 0.7]).all():
        color = 'm'
    elif (ep_trajectory[-1, 3:5] == [0.7, 0.3]).all():
        color = 'k'

    ep_trajectory[:, 0] = ((np.asarray(ep_trajectory[:, 0]) + 1) / 2) * 500
    ep_trajectory[:, 1] = ((np.asarray(ep_trajectory[:, 1]) + 1) / 2) * 500

    # if thr == None:
    #     thr = np.mean(diff)
    #
    # for point, point_n, d in zip(ep_trajectory[:-1], ep_trajectory[1:], diff):
    #     if d > thr:
    #         plt.plot([point[0], point_n[0]], [point[1], point_n[1]], 'r')
    #     else:
    #         plt.plot([point[0], point_n[0]], [point[1], point_n[1]], color)
    plt.plot(ep_trajectory[:, 0], ep_trajectory[:, 1], color)


if __name__ == '__main__':

    # Open the trajectories file, if exists. A trajectory is a list of points (& inventory) encountered during training
    trajectories = None
    try:
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
        print(len(trajectories))
        # do your stuff
        # with open("arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
        #     trajectories = json.load(f)

    except Exception as e:
        print("traj problem")
        print(e)
        pass

    if trajectories == None:
        try:
            with open("arrays/{}.pickle".format("{}_trajectories".format(model_name)), 'rb') as f:
                trajectories = pickle.load(f)
        except Exception as e:
            print("traj problem")
            print(e)
            pass

    # As well as the action made in the episode
    # actions = None
    # try:
    #     with open("arrays/{}.json".format("{}_actions".format(model_name))) as f:
    #         actions = json.load(f)
    # except Exception as e:
    #     print("act problem")
    #     print(e)
    #     pass

    # Create pos_buffer from trajectories
    world_model = []
    buffer = trajectories_to_pos_buffer(trajectories, world_model)
    world_model = np.asarray(world_model)
    # plt.figure()
    world_model[:,3] = np.clip(world_model[:,3], 0, np.max(world_model[:,3]/550))
    view, canvas = plot_3d_map(world_model)
    # app.run()
    # input('...')
    # plt.show()

    # Create Heatmap
    heatmap = np.zeros((800, 800))
    covmap = np.zeros((800, 800))
    # graph = dict(x=[], y=[], z=[])
    for k in buffer.keys():

        k_value = list(map(float, k.split(" ")))
        k_value = np.asarray(k_value).astype(int)
        try:
            heatmap[k_value[0], k_value[1]] += buffer[k]
            covmap[k_value[0], k_value[1]] = 1
            # graph['x'].append(k_value[0])
            # graph['z'].append(k_value[1])
            # graph['y'].append(k_value[2])

        except Exception as e:
            print(k)
            input('...')

    # json_str = json.dumps(graph, cls=NumpyEncoder)
    # f = open("../OpenWorldEnv/OpenWorld/Assets/Resources/graph.json".format(model_name), "w")
    # f.write(json_str)
    # f.close()

    heatmap = np.clip(heatmap, 0, np.max(heatmap)/50)

    heatmap = np.rot90(heatmap)
    covmap = np.rot90(covmap)

    # # Plot heatmap
    # plt.figure()
    # plot_map(heatmap)
    #
    # # Plot coverage map
    # plt.figure()
    # plot_map(covmap)

    # Compute the cumulative reward of the learnt RND and GAIL to compare trajectories
    if trajectories is not None and actions is not None:

        graph = tf.compat.v1.Graph()
        motivation = None
        reward_model = None
        try:
            # Load motivation model
            with graph.as_default():
                # model_name = "asdasdasd"
                tf.compat.v1.disable_eager_execution()
                motivation_sess = tf.compat.v1.Session(graph=graph)
                motivation = RND(motivation_sess, input_spec=input_spec, network_spec_predictor=network_spec_rnd_predictor,
                                 network_spec_target=network_spec_rnd_target, obs_normalization=False,
                                 obs_to_state=obs_to_state_rnd, motivation_weight=1)
                init = tf.compat.v1.global_variables_initializer()
                motivation_sess.run(init)
                motivation.load_model(name=model_name, folder='saved')

            # Load imitation model
            # graph = tf.compat.v1.Graph()
            # with graph.as_default():
            #     from architectures.bug_arch_very_acc import *
            #     model_name = 'vaffanculo_im'
            #     reward_model_name = "vaffanculo_9000"
            #     tf.compat.v1.disable_eager_execution()
            #     reward_sess = tf.compat.v1.Session(graph=graph)
            #     reward_model = GAIL(input_architecture=input_spec_irl, network_architecture=network_spec_irl,
            #                         obs_to_state=obs_to_state_irl, actions_size=9, policy=None, sess=reward_sess,
            #                         lr=7e-5, reward_model_weight=0.7,
            #                         name=model_name, fixed_reward_model=False, with_action=True)
            #     init = tf.compat.v1.global_variables_initializer()
            #     reward_sess.run(init)
            #     reward_model.load_model(reward_model_name)
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
            desired_point_y = 28
            goal_area_x = 35
            goal_area_z = 18
            goal_area_y = 28
            goal_area_height = 44
            goal_area_width = 44

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
            sum_moti_rews = []
            sum_moti_rews_dict = dict()
            sum_il_rews = []
            moti_rews = []
            points = []

            step_moti_rews = []
            step_il_rews = []

            max_length = -9999
            mean_length = 0

            pos_buffer = dict()
            # Get only those trajectories that touch the desired points
            for keys, traj in zip(list(trajectories.keys())[:30000], list(trajectories.values())[:30000]):
                # to_observe = False
                # for point in traj:
                #     de_point = np.zeros(3)
                #     de_point[0] = ((np.asarray(point[0]) + 1) / 2) * 500
                #     de_point[1] = ((np.asarray(point[1]) + 1) / 2) * 500
                #     de_point[2] = ((np.asarray(point[2]) + 1) / 2) * 40
                #     if np.abs(de_point[0] - 10) < threshold and \
                #             np.abs(de_point[1] - 463) < threshold and \
                #             np.abs(de_point[2] - 40) < threshold:
                #         to_observe = True
                #         break
                #
                # if to_observe:
                    for i, point in enumerate(traj):
                        de_point = np.zeros(3)
                        de_point[0] = ((np.asarray(point[0]) + 1) / 2) * 500
                        de_point[1] = ((np.asarray(point[1]) + 1) / 2) * 500
                        de_point[2] = ((np.asarray(point[2]) + 1) / 2) * 60
                #         # if np.abs(de_point[0] - desired_point_x) < threshold and \
                #         #         np.abs(de_point[1] - desired_point_z) < threshold :
                        if goal_area_x < de_point[0] < (goal_area_x + goal_area_width) and \
                                 goal_area_z < de_point[1] < (goal_area_z + goal_area_height) and \
                                    np.abs(de_point[2] - desired_point_y) < threshold:# and \
                                       # point[-1] <= 0.5:
                #         if True:
                            traj_len = len(traj)
                            traj_to_observe.append(traj)
                            episodes_to_observe.append(keys)

                            #print(keys)
                            # for j in range(i + 1, traj_len):
                            #     traj[j] = traj[i]

                            # for pos_point in traj:
                            #     insert_to_pos_table(pos_buffer, np.asarray(pos_point[:3]), 1 / 40)
                            break

            # Cluster trajectories to reduce the number of trajectories to observe
            # traj_to_observe = np.asarray(traj_to_observe)
            # with open('traj_to_observe.npy', 'wb') as f:
            #     np.save(f, traj_to_observe)
            # input('...')

            # cluster_indices = cluster(traj_to_observe, 'clustering/autoencoders/jump')
            # cluster_indices = [4229,  239, 9062, 6959, 7693, 2169,  389,  153,   28, 2475]
            # traj_to_observe = traj_to_observe[cluster_indices]
            # new_episode_to_observe = []
            # for id in cluster_indices:
            #     new_episode_to_observe.append(episodes_to_observe[id])
            # episodes_to_observe = new_episode_to_observe

            # Get the value of the motivation and imitation models of the extracted trajectories
            for key, traj, idx_traj in zip(episodes_to_observe, traj_to_observe, range(len(traj_to_observe))):
                states_batch = []
                actions_batch = []

                for state, action in zip(traj, actions[key]):
                    # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
                    # TODO: correct form
                    state = np.asarray(state)
                    #state[:3] = 2 * (state[:3]/40) - 1
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

                    # pos_key = ' '.join(map(str, state[:3]))
                    # counter = pos_buffer[pos_key]
                    # moti_rew = compute_intrinsic_reward(counter)
                    # moti_rews.append(moti_rew)
                    # step_moti_rews.extend(moti_rew)

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

                #il_rew = reward_model.eval(states_batch[:-1], states_batch, actions_batch)
                il_rew = np.zeros(len(states_batch[:-1]))
                step_il_rews.extend(il_rew)
                il_rew = np.sum(il_rew)
                sum_il_rews.append(il_rew)

                moti_rew = motivation.eval(states_batch)
                moti_rews.append(moti_rew)
                step_moti_rews.extend(moti_rew)
                points.extend([k['global_in'] for k in states_batch])
                moti_rew = np.mean(moti_rew)
                sum_moti_rews.append(moti_rew)
                sum_moti_rews_dict[idx_traj] = moti_rew


            # # Try to print points that have high value of IM
            # indices = np.where(step_moti_rews > np.asarray(0.1))
            # indices = np.reshape(indices, -1)
            # points = np.asarray(points)
            # print(np.shape(points))
            # points_to_plot = points[indices]
            # points_to_plot = dict(x=points_to_plot[:, 0], z=points_to_plot[:, 1], y=points_to_plot[:, 2])
            # json_str = json.dumps(points_to_plot, cls=NumpyEncoder)
            # f = open("../OpenWorldEnv/OpenWorld/Assets/Resources/graph.json".format(model_name), "w")
            # f.write(json_str)
            # f.close()

            moti_mean = np.mean(sum_moti_rews)
            il_mean = np.mean(sum_il_rews)
            moti_max = np.max(sum_moti_rews)
            moti_min = np.min(sum_moti_rews)
            il_max = np.max(sum_il_rews)
            il_min = np.min(sum_il_rews)
            epsilon = 0.05
            print(np.max(sum_moti_rews))
            print(np.max(sum_il_rews))
            print(np.median(sum_il_rews))
            print(np.median(moti_mean))
            print(moti_mean)
            print(il_mean)
            print(np.min(sum_il_rews))
            print(np.min(sum_moti_rews))
            print(" ")
            print("Min step moti: {}".format(np.min(step_moti_rews)))
            print("Min step IL: {}".format(np.min(step_il_rews)))
            print("Max step moti: {}".format(np.max(step_moti_rews)))
            print("Max step IL: {}".format(np.max(step_il_rews)))
            print("Mean step moti: {}".format(np.mean(step_moti_rews)))
            print("Mean step IL: {}".format(np.mean(step_il_rews)))
            print(" ")

            # Get those trajectories that have an high motivation reward AND a low imitation reward
            sum_moti_rews_dict = {k: v for k, v in sorted(sum_moti_rews_dict.items(), key=lambda item: item[1], reverse=True)}
            moti_to_observe = [k for k in sum_moti_rews_dict.keys()]
            # moti_to_observe = []
            # for k, v in zip(sum_moti_rews_dict.keys(), sum_moti_rews_dict.values()):
            #     if v > 0.03:
            #         moti_to_observe.append(k)
            moti_to_observe = np.reshape(moti_to_observe, -1)

            il_to_observe = np.where(sum_il_rews > np.asarray(il_mean))
            il_to_observe = np.reshape(il_to_observe, -1)
            idxs_to_observe, idxs1, idxs2 = np.intersect1d(moti_to_observe, il_to_observe, return_indices=True)
            idxs_to_observe = moti_to_observe[np.sort(idxs1)]
            traj_to_observe = np.asarray(traj_to_observe)

            idxs_to_observe = moti_to_observe
            print(moti_to_observe)
            print(idxs_to_observe)

            print("The bugged trajectories are {}".format(len(idxs_to_observe)))

            all_normalized_im_rews = []
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
                im_rew = savitzky_golay(im_rew, 51, 3)
                im_rew = (im_rew - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))
                all_normalized_im_rews.append(im_rew)

            if False:
            # if len(all_normalized_im_rews) > 20:
                cluster_indices = cluster(all_normalized_im_rews, clusters=20)
            else:
                cluster_indices = np.arange(len(all_normalized_im_rews))

            new_episode_to_observe = []
            episodes_to_observe = np.asarray(episodes_to_observe)[idxs_to_observe][cluster_indices]
            all_normalized_im_rews = np.asarray(all_normalized_im_rews)

            # fig = plt.figure()

            # # Aggregated trajectory
            # # Create cells
            # width = height = 500
            # jump = 60
            # agg_width = agg_height = 20
            # agg_jump = 5
            #
            # cells = []
            # for i in range(0, width, agg_width):
            #     for j in range(0, height, agg_height):
            #         for z in range(0, jump, agg_jump):
            #             cells.append((agg_width / 2 + i, agg_height / 2 + j, agg_jump / 2 + z))

            thr = np.mean(step_moti_rews)
            for i, traj, im_rews, key in zip(range(len(cluster_indices)), traj_to_observe[idxs_to_observe][cluster_indices],
                                        all_normalized_im_rews[cluster_indices], episodes_to_observe):
                print_3d_traj(traj, im_rews, view, canvas, actions[key])
            app.run()
            input('...')
            # goal_region = patches.Rectangle((goal_area_x, goal_area_z), goal_area_width, goal_area_height, linewidth=5, edgecolor='r',
            #                                 facecolor='none', zorder=100)
            # fig.gca().add_patch(goal_region)
            # plt.show()
            traj_to_observe = traj_to_observe[idxs_to_observe][cluster_indices]



            for traj, key in zip(traj_to_observe, episodes_to_observe):
                states_batch = []
                actions_batch = []
                #key = episodes_to_observe[idx]

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
                plt.figure()
                plt.title("im: {}".format(np.sum(im_rew)))
                im_rew = savitzky_golay(im_rew, 51, 3)
                im_rew = (im_rew - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))

                print(key)
                all_normalized_im_rews.append(im_rew)
                # plt.plot(range(len(im_rew)), im_rew)
                #plt.plot(range(len(im_rew)), diff)
                # plt.legend(['irl', 'im', 'diff'])

                # TODO: save actions and trajectories, temporarely
                actions_to_save = dict(actions=actions[key])
                json_str = json.dumps(actions_to_save, cls=NumpyEncoder)
                f = open("arrays/actions.json".format(model_name), "w")
                f.write(json_str)
                f.close()

                traj_to_save = dict(x_s=traj[:, 0], z_s=traj[:, 1], y_s=traj[:, 2], im_values=im_rew,
                                    il_values=np.zeros(len(states_batch)))
                json_str = json.dumps(traj_to_save, cls=NumpyEncoder)
                f = open("../Playtesting-Env/Assets/Resources/traj_{}.json".format(key), "w")
                f.write(json_str)
                f.close()

                # fig = plt.figure()
                # print_traj_with_diff(traj, im_rew, thr)
                # goal_region = patches.Rectangle((goal_area_x, goal_area_z), goal_area_width, goal_area_height,
                #                                 linewidth=5, edgecolor='r',
                #                                 facecolor='none', zorder=100)
                # fig.gca().add_patch(goal_region)
                #
                # plt.show()
                # plt.waitforbuttonpress()

        plt.show()
