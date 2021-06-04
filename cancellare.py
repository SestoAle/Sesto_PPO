import pickle
import json
import numpy as np

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

    model_name = 'bug_detector_gail_schifo_acc_com_irl_im_3_no_key_5_2'

    expert_traj = load_demonstrations('dems_acc_com_no_key_muted.pkl')
    with open("arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
        saved_trajectories = json.load(f)

    with open("arrays/{}.json".format("{}_actions".format(model_name))) as f:
        saved_actions = json.load(f)

    print('Demonstrations loaded! We have ' + str(len(expert_traj['obs'])) + " timesteps in these demonstrations")

    input('...')
    expert_traj = saved_trajectories_to_demonstrations([saved_trajectories['0']], [saved_actions['0']], expert_traj)

    save_demonstrations(expert_traj, name='dems_acc_com_no_key_muted.pkl')
    print('Demonstrations loaded! We have ' + str(len(expert_traj['obs'])) + " timesteps in these demonstrations")