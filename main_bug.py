from agents.PPO import PPO
from architectures.bug_arch import *
from runner.runner import Runner
from runner.parallel_runner import Runner as ParallelRunner
from motivation.random_network_distillation import RND
import os
import tensorflow as tf
import argparse
import numpy as np
import math
import gym
# Load UnityEnvironment and my wrapper
from mlagents.envs import UnityEnvironment
import logging as logs

from reward_model.reward_model import GAIL

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='bug_detector_gail_schifo_irl')
parser.add_argument('-gn', '--game-name', help="The name of the game", default=None)
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-sf', '--save-frequency', help="How mane episodes after save the model", default=3000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=100)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)
parser.add_argument('-rc', '--recurrent', dest='recurrent', action='store_true')
parser.add_argument('-pl', '--parallel', dest='parallel', action='store_true')

# Parse arguments for Inverse Reinforcement Learning
parser.add_argument('-irl', '--inverse-reinforcement-learning', dest='use_reward_model', action='store_true')
parser.add_argument('-rf', '--reward-frequency', help="How many episode before update the reward model", default=1)
parser.add_argument('-rm', '--reward-model', help="The name of the reward model", default='bug_detector_gail_schifo_irl_3000')
parser.add_argument('-dn', '--dems-name', help="The name of the demonstrations file", default='dems.pkl')
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

    def execute(self, actions):
        #actions = int(input(': '))

        env_info = self.unity_env.step([actions])[self.default_brain]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        self.previous_action = actions

        state = dict(global_in=env_info.vector_observations[0])

        # Get the agent position from the state to compute reward
        position = state['global_in'][:3]

        # Get the counter of that position and compute reward
        counter = self.insert_to_pos_table(position)
        # reward = self.compute_intrinsic_reward(counter)
        # reward = 0

        # print(state['global_in'][:2])
        # print(np.flip(np.transpose(np.reshape(state['global_in'][7:7+25], [5,5])), 0))
        # print(state['global_in'][7:7+12])
        # print(reward)
        # print(done)

        return state, done, reward

    def reset(self):

        self.previous_action = [0, 0]
        logs.getLogger("mlagents.envs").setLevel(logs.WARNING)
        self.coverage_of_points.append(len(env.pos_buffer.keys()))
        # self.set_spawn_position()

        env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
        state = dict(global_in=env_info.vector_observations[0])
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

    # Compute the intrinsic reward based on the counter
    def compute_intrinsic_reward(self, counter):
        return self.r_max * (1 - (counter / self.max_counter))

def callback(agent, env, runner):
    positions = len(env.pos_buffer.keys())
    print('Coverage of points: {}'.format(positions))

    # Save position buffer as json
    json_str = json.dumps(env.pos_buffer, cls=NumpyEncoder)
    f = open("arrays/{}_pos_buffer.json".format(model_name), "w")
    f.write(json_str)
    f.close()

    # Save position coverage during time as json
    json_str = json.dumps(env.coverage_of_points, cls=NumpyEncoder)
    f = open("arrays/{}_coverage.json".format(model_name), "w")
    f.write(json_str)
    f.close()


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
    # IRL
    use_reward_model = args.use_reward_model
    reward_model_name = args.reward_model
    fixed_reward_model = args.fixed_reward_model
    dems_name = args.dems_name
    reward_frequency = int(args.reward_frequency)

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = {
        'current_step': 0,
        "thresholds": [1000000000000000000000000000],
        "parameters": {
            "agent_spawn_x": [14],
            "agent_spawn_z": [14]
        }
    }

    # Total episode of training
    total_episode = 1e10
    # Units of training (episodes or timesteps)
    frequency_mode = 'episodes'
    # Frequency of training (in episode)
    frequency = 5
    # Memory of the agent (in episode)
    memory = 5

    # Create agent
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session(graph=graph)
        agent = PPO(sess, input_spec=input_spec, network_spec=network_spec, obs_to_state=obs_to_state,
                    action_type='discrete', action_size=9, model_name=model_name, p_lr=7e-5, v_batch_fraction=1.,
                    v_num_itr=1, memory=memory,
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
            motivation = RND(motivation_sess, input_spec=input_spec, network_spec=network_spec_rnd,
                             obs_to_state=obs_to_state_rnd)
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
                                name=model_name, fixed_reward_model=False, with_action=True)
            init = tf.compat.v1.global_variables_initializer()
            reward_sess.run(init)
            # If we want, we can use an already trained reward model
            if fixed_reward_model:
                reward_model.load_model(reward_model_name)
                print("Model loaded!")


    # Open the environment with all the desired flags
    if not parallel:
        # Open the environment with all the desired flags
        env = BugEnvironment(game_name=game_name, no_graphics=True, worker_id=work_id, max_episode_timesteps=max_episode_timestep)
    else:
        # If parallel, create more environments
        envs = []
        for i in range(5):
            envs.append(BugEnvironment(game_name=game_name, no_graphics=True, worker_id=work_id + i,
                                       max_episode_timesteps=max_episode_timestep))

    # Create runner
    if not parallel:
        runner = Runner(agent=agent, frequency=frequency, env=env, save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum,
                        frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback,
                        reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                        fixed_reward_model=fixed_reward_model, motivation=motivation)
    else:
        runner = ParallelRunner(agent=agent, frequency=frequency, envs=envs, save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum,
                        frequency_mode=frequency_mode, curriculum_mode='episodes', callback_function=callback,
                        reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                        fixed_reward_model=fixed_reward_model)

    try:
        runner.run()
    finally:
        if not parallel:
            env.close()
        else:
            for env in envs:
                env.close()
