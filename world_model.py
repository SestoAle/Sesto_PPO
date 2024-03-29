import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from math import factorial
import tensorflow as tf
from motivation.random_network_distillation import RND
from reward_model.reward_model import GAIL
from architectures.bug_arch_complex import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

name1 = "bug_detector_gail_schifo_3"
name2 = "bug_detector_gail_schifo_moti"
name3 = "bug_detector_gail_schifo_irl"

name4 = 'bug_detector_gail_schifo_complex'
name5 = 'bug_detector_gail_schifo_complex_irl_moti_2'
name6 = 'bug_detector_gail_schifo_complex_moti_3'

model_name = name5

reward_model_name = None
if model_name == name5:
    reward_model_name = "bug_detector_gail_schifo_complex_irl_moti_2_21000"

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

    ax.set_xticks(np.arange(heatmap.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(heatmap.shape[0] + 1) - .5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)


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

def print_traj(traj):
    """
    Method that will plot the trajectory
    """
    ep_trajectory = np.asarray(traj)
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    color = 'g'

    if (ep_trajectory[-1, 3:5] == [0, 0]).all():
        color = 'g'
    elif (ep_trajectory[-1, 3:5] == [0, 1]).all():
        color = 'b'
    elif (ep_trajectory[-1, 3:5] == [1, 0]).all():
        color = 'y'
    elif (ep_trajectory[-1, 3:5] == [1, 1]).all():
        color = 'm'
    plt.plot(ep_trajectory[:, 0], ep_trajectory[:, 1], color)

def print_traj_with_diff(traj, diff):
    """
    Method that will plot the trajectory
    """
    ep_trajectory = np.asarray(traj)
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    color = 'g'

    if (ep_trajectory[-1, 3:5] == [0, 0]).all():
        color = 'g'
    elif (ep_trajectory[-1, 3:5] == [0, 1]).all():
        color = 'b'
    elif (ep_trajectory[-1, 3:5] == [1, 0]).all():
        color = 'y'
    elif (ep_trajectory[-1, 3:5] == [1, 1]).all():
        color = 'm'

    ep_trajectory = ((np.asarray(ep_trajectory) + 1) / 2) * 40

    for point, point_n, d in zip(ep_trajectory[:-1], ep_trajectory[1:], diff):
        if d < 0.25:
            plt.plot([point[0], point_n[0]], [point[1], point_n[1]], 'red')
        else:
            plt.plot([point[0], point_n[0]], [point[1], point_n[1]], color)

if __name__ == '__main__':

    # Open the position buffer file
    with open("arrays/{}.json".format("{}_pos_buffer".format(model_name))) as f:
        buffer = json.load(f)

    # Open the coverage during time file
    with open("arrays/{}.json".format("{}_coverage".format(model_name))) as f:
        coverage = json.load(f)

    # Open the trajectories file, if exists. A trajectory is a list of points (& inventory) encountered during training
    trajectories = None
    try:
        with open("arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
            trajectories = json.load(f)

    except Exception as e:
        print(e)
        pass

    # As well as the action made in the episode
    actions = None
    try:
        with open("arrays/{}.json".format("{}_actions".format(model_name))) as f:
            actions = json.load(f)

    except Exception as e:
        print(e)
        pass

    # Create Heatmap
    heatmap = np.zeros((40, 40))
    covmap = np.zeros((40, 40))
    for k in buffer.keys():
        k_value = list(map(float, k.split(" ")))
        k_value = np.asarray(k_value)
        k_value = (((k_value + 1) / 2) * 39)
        k_value = k_value.astype(int)

        heatmap[k_value[0], k_value[1]] += buffer[k]
        covmap[k_value[0], k_value[1]] = 1

    heatmap = np.clip(heatmap, 0, np.max(heatmap) / 5)

    heatmap = np.rot90(heatmap)
    covmap = np.rot90(covmap)

    # Plot heatmap
    plt.figure()
    plot_map(heatmap)

    # Plot coverage map
    plt.figure()
    plot_map(covmap)

    # Show coverage during time
    fig = plt.figure()
    plt.plot(range(len(coverage)), coverage)

    # Compute the cumulative reward of the learnt RND and GAIL to compare trajectories

    if trajectories is not None and actions is not None:

        graph = tf.compat.v1.Graph()
        motivation = None
        reward_model = None
        try:
            # Load motivation model
            with graph.as_default():
                tf.compat.v1.disable_eager_execution()
                motivation_sess = tf.compat.v1.Session(graph=graph)
                motivation = RND(motivation_sess, input_spec=input_spec, network_spec=network_spec_rnd,
                                 obs_to_state=obs_to_state_rnd)
                init = tf.compat.v1.global_variables_initializer()
                motivation_sess.run(init)
                motivation.load_model(name=model_name, folder='saved')

            # Load imitation model
            with graph.as_default():
                tf.compat.v1.disable_eager_execution()
                reward_sess = tf.compat.v1.Session(graph=graph)
                reward_model = GAIL(input_architecture=input_spec_irl, network_architecture=network_spec_irl,
                                    obs_to_state=obs_to_state_irl, actions_size=9, policy=None, sess=reward_sess,
                                    lr=7e-5,
                                    name=model_name, fixed_reward_model=False, with_action=True)
                init = tf.compat.v1.global_variables_initializer()
                reward_sess.run(init)
                reward_model.load_model(reward_model_name)
        except Exception as e:
            reward_model = None
            print(e)

        if motivation is not None and reward_model is not None:

            # Filler the state
            # TODO: I do this because the state that I saved is only the points AND inventory, not the complete state
            # TODO: it is probably better to save everything
            filler = np.zeros((66))
            filler[-2] = 1.
            traj_to_observe = []
            episodes_to_observe = []

            # Define the desired points to check
            # I will get all the saved trajectories that touch one of these points at least once
            desired_point_x = 35
            desired_point_z = 5
            threshold = 5

            # Save the motivation rewards and the imitation rewards
            moti_rews = []
            il_rews = []

            # Get only those trajectories that touch the desired points
            for keys, traj in zip(trajectories.keys(), trajectories.values()):
                for point in traj:
                    if np.abs(point[0] - desired_point_x) < threshold and np.abs(point[1] - desired_point_z) < threshold:
                        traj_to_observe.append(traj)
                        episodes_to_observe.append(keys)
                        break

            # Get the value of the motivation and imitation models of the extracted trajectories
            for key, traj in zip(episodes_to_observe, traj_to_observe):
                states_batch = []
                actions_batch = []
                for state in traj:
                    # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
                    # TODO: correct form
                    state = np.asarray(state)
                    state[:3] = 2 * (state[:3]/40) - 1
                    state = np.concatenate([state, filler])
                    state[-2:] = state[3:5]

                    # Create the states batch to feed the models
                    state = dict(global_in=state)
                    states_batch.append(state)

                # Get the action batches for imitation model
                for action in actions[key]:
                    actions_batch.append(action)

                il_rew = np.sum(reward_model.eval(states_batch, states_batch, actions_batch))
                il_rews.append(il_rew)

                moti_rew = np.sum(motivation.eval(states_batch))
                moti_rews.append(moti_rew)

            moti_mean = np.mean(moti_rews)
            il_mean = np.mean(il_rews)
            moti_max = np.max(moti_rews)
            moti_min = np.min(moti_rews)
            il_max = np.max(il_rews)
            il_min = np.min(il_rews)
            epsilon = 0.5
            print(np.max(moti_rews))
            print(np.max(il_rews))
            print(np.median(il_rews))
            print(np.median(moti_mean))
            print(moti_mean)
            print(il_mean)
            print(np.min(il_rews))

            # Get those trajectories that have an high motivation reward AND a low imitation reward
            moti_to_observe = np.where(moti_rews > np.asarray(8))
            moti_to_observe = np.reshape(moti_to_observe, -1)
            il_to_observe = np.where(il_rews < il_min + epsilon)
            il_to_observe = np.reshape(il_to_observe, -1)
            idxs_to_observe = np.intersect1d(il_to_observe, moti_to_observe)
            traj_to_observe = np.asarray(traj_to_observe)

            # Plot the trajectories
            for traj, idx in zip(traj_to_observe[moti_to_observe], moti_to_observe):

                # Plot models values
                states_batch = []

                key = episodes_to_observe[idx]

                for state in traj:
                    # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
                    # TODO: correct form
                    state = np.asarray(state)
                    state[:3] = 2 * (state[:3] / 40) - 1

                    state = np.concatenate([state, filler])

                    state[-2:] = state[3:5]

                    # Create the states batch to feed the models
                    state = dict(global_in=state)
                    states_batch.append(state)

                actions_batch = []
                for action in actions[key]:
                    actions_batch.append(action)

                irl_rew = reward_model.eval(states_batch, states_batch, actions_batch)
                im_rew = motivation.eval(states_batch)

                irl_rew = savitzky_golay(irl_rew, 51, 3)
                im_rew = savitzky_golay(im_rew, 51, 3)

                irl_rew = (irl_rew - np.min(irl_rew)) / (np.max(irl_rew) - np.min(irl_rew))
                im_rew = (im_rew - np.min(im_rew)) / (np.max(im_rew) - np.min(im_rew))

                plt.figure()
                plt.plot(range(len(irl_rew)), irl_rew)
                plt.plot(range(len(im_rew)), im_rew)
                plt.legend(['irl rew', 'il rew'])

                plt.figure()
                print_traj_with_diff(traj, irl_rew)

                plt.show()
                plt.waitforbuttonpress()

    plt.show()