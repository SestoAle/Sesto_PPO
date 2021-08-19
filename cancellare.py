import pickle
import json
import numpy as np
from utils import NumpyEncoder
from clustering.clustering_ae import cluster

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

def saved_trajectories_to_demonstrations(trajectories, actions, demonstrations):
    '''
    This method will take some trajectories saved from Intrinsic Motivation + Imitation Learning training
    and transform it into a demonstrations that can be used with Imitation Learning.
    TODO: This method is valid only with the current world_model_3d.py script
    '''

    filler = np.zeros(67)
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

if __name__ == '__main__':
    #trajectories = np.load('traj_to_observe.npy')
    #cluster(trajectories, 'clustering/autoencoders/labyrinth')
    # model_name = 'bug_detector_gail_schifo_acc_com_irl_im_3_no_key_5_2'
    #
    expert_traj = load_demonstrations('dem_acc_really_big_only_jump_3d_v9.pkl')
    expert_traj = [state['global_in'] for state in expert_traj['obs']]
    expert_traj = np.asarray(expert_traj)

    mean_x = 0
    mean_y = 0
    mean_z = 0

    for i in range(715, 716):
        mean_x += expert_traj[i, 0]
        mean_y += expert_traj[i, 1]
        mean_z += expert_traj[i, 2]

    mean_x /= 1
    mean_y /= 1
    mean_z /= 1

    mean_x = ((mean_x + 1) / 2) * 220
    mean_y = ((mean_y + 1) / 2) * 280
    mean_z = ((mean_z + 1) / 2) * 40

    print(mean_x)
    print(mean_y)
    print(mean_z)



    # second_expert_traj = load_demonstrations('dem_acc_impossibru_to_add.pkl')
    #
    # expert_traj['obs'].extend(second_expert_traj['obs'])
    # expert_traj['obs_n'].extend(second_expert_traj['obs_n'])
    # expert_traj['acts'].extend(second_expert_traj['acts'])
    #
    # save_demonstrations(expert_traj, name='dem_acc_impossibru_both.pkl')
    #
    # input('...')

    # with open("arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
    #     saved_trajectories = json.load(f)
    #
    # with open("arrays/{}.json".format("{}_actions".format(model_name))) as f:
    #     saved_actions = json.load(f)

    # print('Demonstrations loaded! We have ' + str(len(expert_traj['obs'])) + " timesteps in these demonstrations")
    # for i in range(8):
    #     traj = []
    #     for j in range(i * 80, (i * 80 + 80)):
    #         traj.append(expert_traj['obs'][j]['global_in'][:3])
    #
    #     traj = np.asarray(traj)
    #     traj_to_save = dict(x_s=traj[:, 0], z_s=traj[:, 1], y_s=traj[:, 2], im_values=np.zeros(80), il_values=np.zeros(80))
    #     json_str = json.dumps(traj_to_save, cls=NumpyEncoder)
    #     f = open("../OpenWorldEnv/OpenWorld/Assets/Resources/traj.json".format(model_name), "w")
    #     f.write(json_str)
    #     f.close()
    #     input('...')
    # expert_traj = saved_trajectories_to_demonstrations([saved_trajectories['0']], [saved_actions['0']], expert_traj)

    # save_demonstrations(expert_traj, name='dems_acc_com_no_key_muted.pkl')
    # print('Demonstrations loaded! We have ' + str(len(expert_traj['obs'])) + " timesteps in these demonstrations")