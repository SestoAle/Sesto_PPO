from agents.PPO import PPO
from architectures.bug_arch_really_acc_3d import *
from runner.runner import Runner
from runner.parallel_runner import Runner as ParallelRunner
from motivation.random_network_distillation import RND
import os
import tensorflow as tf
import argparse
import pickle
import numpy as np
import math
import gym
# Load UnityEnvironment and my wrapper
from mlagents.envs import UnityEnvironment
import matplotlib.pyplot as plt
import logging as logs

from reward_model.reward_model import GAIL

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='questoeimpossibile_rc')
parser.add_argument('-gn', '--game-name', help="The name of the game", default=None)
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-sf', '--save-frequency', help="How mane episodes after save the model", default=3000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=320)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)
parser.add_argument('-rc', '--recurrent', dest='recurrent', action='store_true')
parser.add_argument('-pl', '--parallel', dest='parallel', action='store_true')
parser.add_argument('-ev', '--evaluation', dest='evaluation', action='store_true')

# Parse arguments for Inverse Reinforcement Learning
parser.add_argument('-irl', '--inverse-reinforcement-learning', dest='use_reward_model', action='store_true')
parser.add_argument('-rf', '--reward-frequency', help="How many episode before update the reward model", default=1)
parser.add_argument('-rm', '--reward-model', help="The name of the reward model", default='vaffanculo_6000')
parser.add_argument('-dn', '--dems-name', help="The name of the demonstrations file", default='dem_acc_really_big_only_jump_3d_v9.pkl')
parser.add_argument('-fr', '--fixed-reward-model', help="Whether to use a trained reward model",
                    dest='fixed_reward_model', action='store_true')
parser.add_argument('-gd', '--get-demonstrations', dest='get_demonstrations', action='store_true')

# Parse arguments for Intrinsic Motivation
parser.add_argument('-m', '--motivation', dest='use_motivation', action='store_true')

parser.set_defaults(use_reward_model=False)
parser.set_defaults(fixed_reward_model=False)
parser.set_defaults(recurrent=False)
parser.set_defaults(parallel=False)
parser.set_defaults(use_motivation=False)
parser.set_defaults(get_demonstrations=False)
parser.set_defaults(evaluation=False)

args = parser.parse_args()

eps = 1e-12

class BugEnvironment:

    def __init__(self, game_name, no_graphics, worker_id, max_episode_timesteps, pos_already_normed=True):
        self.no_graphics = no_graphics
        self.unity_env = UnityEnvironment(game_name, no_graphics=no_graphics, seed=worker_id, worker_id=worker_id)
        self._max_episode_timesteps = max_episode_timesteps
        self.default_brain = self.unity_env.brain_names[0]
        self.config = None
        self.actions_eps = 0.1
        self.previous_action = [0, 0]
        # Table where we save the position for intrisic reward and spawn position
        self.pos_buffer = dict()
        self.pos_already_normed = pos_already_normed
        self.r_max = 0.5
        self.max_counter = 500
        self.tau = 1 / 40
        self.standard_position = [14, 14, 1]
        self.coverage_of_points = []

        # Dict to store the trajectories at each episode
        self.trajectories_for_episode = dict()
        # Dict to store the actions at each episode
        self.actions_for_episode = dict()
        self.episode = -1

        # Defined the values to sample for goal-conditioned policy
        self.reward_weights = [0, 0, 0.3, 0.5, 0.7, 1, 1]

    def execute(self, actions):
        # actions = 99
        # while(actions == 99):
        #     actions = self.command_to_action(input(': '))

        env_info = self.unity_env.step([actions])[self.default_brain]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        self.actions_for_episode[self.episode].append(actions)

        self.previous_action = actions

        state = dict(global_in=env_info.vector_observations[0])
        # Append the value of the motivation weight
        state['global_in'] = np.concatenate([state['global_in'], self.sample_weights])

        # Get the agent position from the state to compute reward
        position = state['global_in'][:3]
        self.trajectories_for_episode[self.episode].append(np.concatenate([position, state['global_in'][-2:]]))
        # Get the counter of that position and compute reward
        # counter = self.insert_to_pos_table(position[:2])
        # reward = self.compute_intrinsic_reward(counter)
        # reward = 0

        # print(state['global_in'][:2])
        # print(np.flip(np.transpose(np.reshape(state['global_in'][10:10+225], [15, 15])), 0))
        # print(np.flip(np.transpose(np.reshape(state['global_in'][10+225:10+225 + 225], [15, 15])), 0))
        # Visualize 3D boxcast
        # threedgrid = np.reshape(state['global_in'][10:10 + 9261], [21, 21, 21])
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # filled = (1 - (threedgrid == 0))
        # cmap = plt.get_cmap("viridis")
        # norm = plt.Normalize(threedgrid.min(), threedgrid.max())
        # ax.voxels(filled, facecolors=cmap(norm(threedgrid)), edgecolor="black")
        # plt.show()
        # plt.waitforbuttonpress()

        # print(state['global_in'][-2:])
        # print(state['global_in'][7:7+12])
        # print(reward)
        # print(done)

        return state, done, reward

    def reset(self):

        # Sample a motivation reward weight
        self.sample_weights = self.reward_weights[np.random.randint(len(self.reward_weights))]
        self.sample_weights = [self.sample_weights, 1-self.sample_weights]

        self.previous_action = [0, 0]
        logs.getLogger("mlagents.envs").setLevel(logs.WARNING)
        self.coverage_of_points.append(len(self.pos_buffer.keys()))
        self.episode += 1
        self.trajectories_for_episode[self.episode] = []
        self.actions_for_episode[self.episode] = []
        # self.set_spawn_position()

        env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
        state = dict(global_in=env_info.vector_observations[0])
        # Append the value of the motivation weight
        state['global_in'] = np.concatenate([state['global_in'], self.sample_weights])
        position = state['global_in'][:3]
        self.trajectories_for_episode[self.episode].append(np.concatenate([position, state['global_in'][-2:]]))
        # print(np.reshape(state['global_in'][7:7 + 225], [15, 15]))
        return state

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += (p * np.log(p))
        return -entr

    def set_config(self, config):
        self.config = config

    def close(self):
        self.unity_env.close()

    def multidimensional_shifting(self, num_samples, sample_size, elements, probabilities):
        # replicate probabilities as many times as `num_samples`
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        # get random shifting numbers & scale them correctly
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - replicated_probabilities
        return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

    # Spawn a position from the buffer
    # If the buffer is empty, spawn at standard position
    def set_spawn_position(self):
        values = self.pos_buffer.values()
        if len(values) > 0:
            values = np.asarray(list(values))

            probs = 1 / values

            for i in range(len(values)):
                pos_key = list(self.pos_buffer.keys())[i]
                pos = np.asarray(list(map(float, pos_key.split(" "))))
                if pos[2] == 0:
                    probs[i] = 0

            probs = probs / np.sum(probs)

            index = self.multidimensional_shifting(1, 1, np.arange(len(probs)), probs)[0][0]

            pos_key = list(self.pos_buffer.keys())[index]
            pos = np.asarray(list(map(float, pos_key.split(" "))))
            # re-normalize pos to world coordinates
            pos = (((pos + 1) / 2) * 40) - 20

            self.config['agent_spawn_x'] = pos[0]
            self.config['agent_spawn_z'] = pos[1]
        else:
            self.config['agent_spawn_x'] = self.standard_position[0]
            self.config['agent_spawn_z'] = self.standard_position[1]

    # Insert to the table. Position must be a 2 element vector
    # Return the counter of that position
    def insert_to_pos_table(self, position):

        # Check if the position is already in the buffer
        for k in self.pos_buffer.keys():
            # If position - k < tau, then the position is already in the buffer
            # Add its counter to one and return it

            # The position are already normalized by the environment
            k_value = list(map(float, k.split(" ")))
            k_value = np.asarray(k_value)
            position = np.asarray(position)

            distance = np.linalg.norm(k_value - position)
            if distance < self.tau:
                self.pos_buffer[k] += 1
                return self.pos_buffer[k]

        pos_key = ' '.join(map(str, position))
        self.pos_buffer[pos_key] = 1
        return self.pos_buffer[pos_key]

    def clear_buffers(self):
        self.trajectories_for_episode = dict()
        self.actions_for_episode = dict()

    def command_to_action(self, command):

        if command == 'w':
            return 3
        if command == 'a':
            return 2
        if command == 's':
            return 4
        if command == 'd':
            return 1

        if command == 'q':
            return 0

        if command == ' ':
            return 9

        return 99

    # Compute the intrinsic reward based on the counter
    def compute_intrinsic_reward(self, counter):
        return self.r_max * (1 - (counter / self.max_counter))

def callback(agent, env, runner):

    global last_key
    save_frequency = 100

    if runner.ep % save_frequency == 0:
        if isinstance(env, list):

            trajectories_for_episode = dict()
            actions_for_episode = dict()

            for e in env:
                for traj, acts in zip(e.trajectories_for_episode.values(), e.actions_for_episode.values()):
                    trajectories_for_episode[last_key] = traj
                    actions_for_episode[last_key] = acts
                    last_key += 1
                e.clear_buffers()
            positions = 0
        else:
            positions = len(env.pos_buffer.keys())
            trajectories_for_episode = env.trajectories_for_episode
            actions_for_episode = env.actions_for_episode

        print('Coverage of points: {}'.format(positions))

        # Save the trajectories
        json_str = json.dumps(trajectories_for_episode, cls=NumpyEncoder)
        f = open("arrays/{}/{}_trajectories_{}.json".format(model_name, model_name, runner.ep), "w")
        f.write(json_str)
        f.close()

        # Save the actions
        json_str = json.dumps(actions_for_episode, cls=NumpyEncoder)
        f = open("arrays/{}/{}_actions_{}.json".format(model_name, model_name, runner.ep), "w")
        f.write(json_str)
        f.close()

        del trajectories_for_episode
        del actions_for_episode


if __name__ == "__main__":

    # DeepCrawl arguments
    model_name = args.model_name
    game_name = args.game_name
    work_id = int(args.work_id)
    save_frequency = int(args.save_frequency)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)
    sampled_env = int(args.sampled_env)
    parallel = args.parallel
    evaluation = args.evaluation
    # IRL
    use_reward_model = args.use_reward_model
    reward_model_name = args.reward_model
    fixed_reward_model = args.fixed_reward_model
    dems_name = args.dems_name
    reward_frequency = int(args.reward_frequency)

    # Central buffer for parallel execution
    if parallel:
        last_key = 0
        if os.path.exists('arrays/{}'.format(model_name)):
            os.rmdir('arrays/{}'.format(model_name))
        os.makedirs('arrays/{}'.format(model_name))
        # trajectories_for_episode = dict()
        # actions_for_episode = dict()
        # # Save the trajectories
        # json_str = json.dumps(trajectories_for_episode, cls=NumpyEncoder)
        # f = open("arrays/{}_trajectories.json".format(model_name), "w")
        # f.write(json_str)
        # f.close()
        #
        # # Save the actions
        # json_str = json.dumps(actions_for_episode, cls=NumpyEncoder)
        # f = open("arrays/{}_actions.json".format(model_name), "w")
        # f.write(json_str)
        # f.close()


    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = {
        'current_step': 0,
        "thresholds": [1000000000000000000000000000],
        "parameters": {
            "agent_spawn_x": [0],
            "agent_spawn_z": [0]
        }
    }

    # Total episode of training
    total_episode = 1e10
    # Units of training (episodes or timesteps)
    frequency_mode = 'episodes'
    # Frequency of training (in episode)
    frequency = 10
    # Memory of the agent (in episode)
    memory = 10

    # Create agent
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session(graph=graph)
        agent = PPO(sess, input_spec=input_spec, network_spec=network_spec, obs_to_state=obs_to_state,
                    action_type='discrete', action_size=10, model_name=model_name, p_lr=7e-5, v_batch_fraction=0.33,
                    v_num_itr=10, memory=memory, c2=0.1,
                    v_lr=7e-5, recurrent=args.recurrent, frequency_mode=frequency_mode, distribution='gaussian',
                    p_num_itr=10, with_circular=True)
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    # If we use intrinsic motivation, create the model
    motivation = None
    if args.use_motivation:
        with graph.as_default():
            tf.compat.v1.disable_eager_execution()
            motivation_sess = tf.compat.v1.Session(graph=graph)
            motivation = RND(motivation_sess, input_spec=input_spec, network_spec_predictor=network_spec_rnd_predictor,
                             network_spec_target= network_spec_rnd_target, lr=7e-5,
                             obs_to_state=obs_to_state_rnd, num_itr=30, motivation_weight=1)
            init = tf.compat.v1.global_variables_initializer()
            motivation_sess.run(init)

    # If we use IRL, create the reward model
    reward_model = None
    if args.use_reward_model:
        with graph.as_default():
            tf.compat.v1.disable_eager_execution()
            reward_sess = tf.compat.v1.Session(graph=graph)
            reward_model = GAIL(input_architecture=input_spec_irl, network_architecture=network_spec_irl,
                                obs_to_state=obs_to_state_irl, actions_size=9, policy=agent, sess=reward_sess, lr=7e-5,
                                name=model_name, fixed_reward_model=False, with_action=True, reward_model_weight=1)
            init = tf.compat.v1.global_variables_initializer()
            reward_sess.run(init)
            # If we want, we can use an already trained reward model
            if fixed_reward_model:
                reward_model.load_model(reward_model_name)
                print("Model loaded!")


    # Open the environment with all the desired flags
    if not parallel:
        # Open the environment with all the desired flags
        env = BugEnvironment(game_name=game_name, no_graphics=True, worker_id=work_id,
                             max_episode_timesteps=max_episode_timestep)
    else:
        # If parallel, create more environments
        envs = []
        for i in range(10):
            envs.append(BugEnvironment(game_name=game_name, no_graphics=True, worker_id=work_id + i,
                                       max_episode_timesteps=max_episode_timestep))

    # Create runner
    if not parallel:
        runner = Runner(agent=agent, frequency=frequency, env=env, save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum,
                        frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback,
                        reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                        fixed_reward_model=fixed_reward_model, motivation=motivation, evaluation=evaluation)
    else:
        runner = ParallelRunner(agent=agent, frequency=frequency, envs=envs, save_frequency=save_frequency,
                                logging=logging, total_episode=total_episode, curriculum=curriculum,
                                frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback,
                                reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                                fixed_reward_model=fixed_reward_model, motivation=motivation, evaluation=evaluation)

    try:
        runner.run()
    finally:
        if not parallel:
            env.close()
        else:
            for env in envs:
                env.close()
