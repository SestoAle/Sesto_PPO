import matplotlib.pyplot as plt
from math import factorial
import os
import pickle
from math import factorial

import matplotlib.pyplot as plt
import numpy as np

from architectures.bug_arch_very_acc import *
from motivation.random_network_distillation import RND
from reward_model.reward_model import GAIL
from clustering.clustering_ae import cluster

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

name_good = 'bug_detector_gail_schifo_acc_com_irl_im_3_no_key_5_2_pl_c2=0.1_replay_random_buffer'

model_name = 'vaffanculo'

reward_model_name = "vaffanculo_im_60000"

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
def trajectories_to_pos_buffer(trajectories, tau=1/40):
    pos_buffer = dict()
    count = 0
    for traj in trajectories.values():
        count += 1
        for state in traj:
            position = np.asarray(state[:2])
            position[0] = (((position[0] + 1) / 2) * 220)
            position[1] = (((position[1] + 1) / 2) * 280)
            position = position.astype(int)
            pos_key = ' '.join(map(str, position))
            if pos_key in pos_buffer.keys():
                pos_buffer[pos_key] += 1
            else:
                pos_buffer[pos_key] = 1
    return pos_buffer


def saved_trajectories_to_demonstrations(trajectories, actions, demonstrations):
    '''
    This method will take some trajectories saved from Intrinsic Motivation + Imitation Learning training
    and transform it into a demonstrations that can be used with Imitation Learning.
    TODO: This method is valid only with the current world_model_3d.py script
    '''

    filler = np.zeros(68)
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

def print_traj(traj):
    """
    Method that will plot the trajectory
    """
    ep_trajectory = np.asarray(traj)
    plt.xlim(0, 220)
    plt.ylim(0, 280)
    color = 'g'

    if (ep_trajectory[-1, 3:5] == [0, 0]).all():
        color = 'g'
    elif (ep_trajectory[-1, 3:5] == [0, 1]).all():
        color = 'b'
    elif (ep_trajectory[-1, 3:5] == [1, 0]).all():
        color = 'y'
    elif (ep_trajectory[-1, 3:5] == [1, 1]).all():
        color = 'm'

    ep_trajectory[:, 0] = ((np.asarray(ep_trajectory[:, 0]) + 1) / 2) * 220
    ep_trajectory[:, 1] = ((np.asarray(ep_trajectory[:, 1]) + 1) / 2) * 280
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
    plt.xlim(0, 220)
    plt.ylim(0, 280)
    color = 'g'

    if (ep_trajectory[-1, 3:5] == [0, 0]).all():
        color = 'g'
    elif (ep_trajectory[-1, 3:5] == [0, 1]).all():
        color = 'b'
    elif (ep_trajectory[-1, 3:5] == [1, 0]).all():
        color = 'y'
    elif (ep_trajectory[-1, 3:5] == [1, 1]).all():
        color = 'm'

    ep_trajectory[:, 0] = ((np.asarray(ep_trajectory[:, 0]) + 1) / 2) * 220
    ep_trajectory[:, 1] = ((np.asarray(ep_trajectory[:, 1]) + 1) / 2) * 280

    if thr == None:
        thr = np.mean(diff)

    for point, point_n, d in zip(ep_trajectory[:-1], ep_trajectory[1:], diff):
        if d > thr:
            plt.plot([point[0], point_n[0]], [point[1], point_n[1]], 'r')
        else:
            plt.plot([point[0], point_n[0]], [point[1], point_n[1]], color)

if __name__ == '__main__':

    # Open the trajectories file, if exists. A trajectory is a list of points (& inventory) encountered during training
    trajectories = None
    try:
        with open("arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
            trajectories = json.load(f)

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
    actions = None
    try:
        with open("arrays/{}.json".format("{}_actions".format(model_name))) as f:
            actions = json.load(f)
    except Exception as e:
        print("act problem")
        print(e)
        pass

    # Create pos_buffer from trajectories
    buffer = trajectories_to_pos_buffer(trajectories)

    # Create Heatmap
    heatmap = np.zeros((320, 480))
    covmap = np.zeros((320, 480))
    for k in buffer.keys():

        k_value = list(map(float, k.split(" ")))
        k_value = np.asarray(k_value).astype(int)
        try:
            heatmap[k_value[0], k_value[1]] += buffer[k]
            covmap[k_value[0], k_value[1]] = 1
        except Exception as e:
            print(k)
            print(2)
            input('...')

    heatmap = np.clip(heatmap, 0, np.max(heatmap)/5)

    heatmap = np.rot90(heatmap)
    covmap = np.rot90(covmap)

    # Plot heatmap
    plt.figure()
    plot_map(heatmap)

    # Plot coverage map
    plt.figure()
    plot_map(covmap)

    # Compute the cumulative reward of the learnt RND and GAIL to compare trajectories
    if trajectories is not None and actions is not None:

        graph = tf.compat.v1.Graph()
        motivation = None
        reward_model = None
        try:
            # Load motivation model
            with graph.as_default():
                #model_name = "double_jump_impossibru_5"
                tf.compat.v1.disable_eager_execution()
                motivation_sess = tf.compat.v1.Session(graph=graph)
                motivation = RND(motivation_sess, input_spec=input_spec, network_spec=network_spec_rnd,
                                 obs_to_state=obs_to_state_rnd, motivation_weight=0.3)
                init = tf.compat.v1.global_variables_initializer()
                motivation_sess.run(init)
                motivation.load_model(name=model_name, folder='saved')

            # Load imitation model
            with graph.as_default():
                model_name = 'vaffanculo_im'
                reward_model_name = "vaffanculo_im_60000"
                tf.compat.v1.disable_eager_execution()
                reward_sess = tf.compat.v1.Session(graph=graph)
                reward_model = GAIL(input_architecture=input_spec_irl, network_architecture=network_spec_irl,
                                    obs_to_state=obs_to_state_irl, actions_size=9, policy=None, sess=reward_sess,
                                    lr=7e-5, reward_model_weight=0.7,
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
            filler = np.zeros((68))
            traj_to_observe = []
            episodes_to_observe = []

            # Define the desired points to check
            # I will get all the saved trajectories that touch one of these points at least once
            desired_point_x = 97
            desired_point_z = 4
            desired_point_y = 1

            threshold = 5

            # Save the motivation rewards and the imitation rewards
            sum_moti_rews = []
            sum_moti_rews_dict = dict()
            sum_il_rews = []
            moti_rews = []

            step_moti_rews = []
            step_il_rews = []

            plt.figure()

            # Get only those trajectories that touch the desired points
            for keys, traj in zip(trajectories.keys(), trajectories.values()):

                # to_observe = False
                # for point in traj:
                #     de_point = np.zeros(2)
                #     de_point[0] = ((np.asarray(point[0]) + 1) / 2) * 100
                #     de_point[1] = ((np.asarray(point[1]) + 1) / 2) * 130
                #     if np.abs(de_point[0] - 55) < threshold and \
                #             np.abs(de_point[1] - 116) < threshold:
                #         to_observe = True
                #         break
                #
                # if to_observe:
                    for point in traj:
                        de_point = np.zeros(3)
                        de_point[0] = ((np.asarray(point[0]) + 1) / 2) * 220
                        de_point[1] = ((np.asarray(point[1]) + 1) / 2) * 280
                        de_point[2] = ((np.asarray(point[2]) + 1) / 2) * 40
                        if np.abs(de_point[0] - desired_point_x) < threshold and \
                                np.abs(de_point[1] - desired_point_z) < threshold: \
                                #and np.abs(de_point[2] - desired_point_y) < threshold:
                            traj_to_observe.append(traj)
                            episodes_to_observe.append(keys)
                            break

            # Cluster trajectories to reduce the number of trajectories to observe
            traj_to_observe = np.asarray(traj_to_observe)
            # with open('traj_to_observe.npy', 'wb') as f:
            #     np.save(f, traj_to_observe)
            # input('...')
            # cluster_indices = cluster(traj_to_observe, 'clustering/autoencoders/jump')
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
                    de_point = np.zeros(2)
                    de_point[0] = ((np.asarray(state['global_in'][0]) + 1) / 2) * 220
                    de_point[1] = ((np.asarray(state['global_in'][1]) + 1) / 2) * 280
                    if np.abs(de_point[0] - desired_point_x) < threshold and \
                            np.abs(de_point[1] - desired_point_z) < threshold:
                        break

                # The actions is one less than the states, so add the last state
                state = traj[-1]
                state = np.concatenate([state, filler])
                state[-2:] = state[3:5]
                state = dict(global_in=state)
                states_batch.append(state)


                il_rew = reward_model.eval(states_batch[:-1], states_batch, actions_batch)
                step_il_rews.extend(il_rew)

                il_rew = np.sum(il_rew)
                sum_il_rews.append(il_rew)
                moti_rew = motivation.eval(states_batch)
                moti_rews.append(moti_rew)
                step_moti_rews.extend(moti_rew)

                moti_rew = np.sum(moti_rew)
                sum_moti_rews.append(moti_rew)
                sum_moti_rews_dict[idx_traj] = moti_rew

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
            # moti_to_observe = np.where(moti_rews > np.asarray(0.30))
            sum_moti_rews_dict = {k: v for k, v in sorted(sum_moti_rews_dict.items(), key=lambda item: item[1], reverse=True)}
            moti_to_observe = [k for k in sum_moti_rews_dict.keys()]
            moti_to_observe = np.reshape(moti_to_observe, -1)

            il_to_observe = np.where(sum_il_rews > np.asarray(il_mean))
            il_to_observe = np.reshape(il_to_observe, -1)
            idxs_to_observe, idxs1, idxs2 = np.intersect1d(moti_to_observe, il_to_observe, return_indices=True)
            idxs_to_observe = moti_to_observe[np.sort(idxs1)]
            traj_to_observe = np.asarray(traj_to_observe)

            idxs_to_observe = moti_to_observe
            print(moti_to_observe)
            print(idxs_to_observe)

            # Plot the trajectories
            plt.figure()
            thr = np.mean(step_moti_rews)
            for i, traj in enumerate(traj_to_observe[idxs_to_observe]):
                print_traj(traj)

            print("The bugged trajectories are {}".format(len(idxs_to_observe)))

            # Increase demonstartions with bugged trajectory
            if False:

                actions_to_save = []
                for i in idxs_to_observe:
                    key = episodes_to_observe[i]
                    actions_to_save.append(actions[key])

                expert_traj = load_demonstrations('dems_acc_com_no_key.pkl')
                with open("arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
                    saved_trajectories = json.load(f)

                with open("arrays/{}.json".format("{}_actions".format(model_name))) as f:
                    saved_actions = json.load(f)

                print('Demonstrations loaded! We have ' + str(
                    len(expert_traj['obs'])) + " timesteps in these demonstrations")

                expert_traj = saved_trajectories_to_demonstrations(traj_to_observe[idxs_to_observe], actions_to_save,
                                                                   expert_traj)

                save_demonstrations(expert_traj, name='dems_acc_com_no_key_muted.pkl')
                print('Demonstrations loaded! We have ' + str(
                    len(expert_traj['acts'])) + " timesteps in these demonstrations")

            # Plot the trajectories
            for traj, idx in zip(traj_to_observe[idxs_to_observe], idxs_to_observe):

                states_batch = []
                key = episodes_to_observe[idx]
                for state in traj:
                    # TODO: In here I will de-normalize and fill the state. Remove this if the states are saved in the
                    # TODO: correct form
                    state = np.asarray(state)
                    # state[:3] = 2 * (state[:3] / 40) - 1
                    state = np.concatenate([state, filler])
                    state[-2:] = state[3:5]

                    # Create the states batch to feed the models
                    state = dict(global_in=state)
                    states_batch.append(state)

                actions_batch = []
                for action in actions[key]:
                    actions_batch.append(action)

                irl_rew = reward_model.eval(states_batch[:-1], states_batch, actions_batch)
                im_rew = motivation.eval(states_batch)
                plt.figure()
                plt.title("im: {}, il: {}".format(np.sum(im_rew), np.sum(irl_rew)))
                irl_rew = savitzky_golay(irl_rew, 51, 3)
                im_rew = savitzky_golay(im_rew, 51, 3)

                irl_rew = (irl_rew - np.min(step_il_rews)) / (np.max(step_il_rews) - np.min(step_il_rews))
                im_rew = (im_rew - np.min(step_moti_rews)) / (np.max(step_moti_rews) - np.min(step_moti_rews))

                # diff = np.asarray(im_rew) - np.asarray(irl_rew)

                plt.plot(range(len(irl_rew)), irl_rew)
                plt.plot(range(len(im_rew)), im_rew)
                #plt.plot(range(len(im_rew)), diff)
                plt.legend(['irl', 'im', 'diff'])

                # TODO: save actions and trajectories, temporarely
                actions_to_save = dict(actions=actions[key])
                json_str = json.dumps(actions_to_save, cls=NumpyEncoder)
                f = open("arrays/actions.json".format(model_name), "w")
                f.write(json_str)
                f.close()

                traj_to_save = dict(x_s=traj[:, 0], z_s=traj[:, 1], y_s=traj[:, 2], im_values=im_rew, il_values=irl_rew)
                json_str = json.dumps(traj_to_save, cls=NumpyEncoder)
                f = open("../OpenWorldEnv/OpenWorld/Assets/Resources/traj.json".format(model_name), "w")
                f.write(json_str)
                f.close()

                plt.figure()
                print_traj_with_diff(traj, im_rew, thr)

                plt.show()
                plt.waitforbuttonpress()

    plt.show()

