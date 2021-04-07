from agents.PPO_openworld import PPO
from runner.runner import Runner
from runner.parallel_runner import Runner as ParallelRunner
import os
import tensorflow as tf
import argparse
import numpy as np
import math
import gym
# Load UnityEnvironment and my wrapper
from mlagents.envs import UnityEnvironment
import logging as logs


from reward_model.reward_model import RewardModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='openworld_discrete_coin3')
parser.add_argument('-gn', '--game-name', help="The name of the game", default="envs/DeepCrawl-Procedural-4")
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-sf', '--save-frequency', help="How mane episodes after save the model", default=3000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=100)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)
parser.add_argument('-rc', '--recurrent', dest='recurrent', action='store_true')
parser.add_argument('-pl', '--parallel', dest='parallel', action='store_true')

# Parse arguments for Inverse Reinforcement Learning
parser.add_argument('-irl', '--inverse-reinforcement-learning', dest='use_reward_model', action='store_true')
parser.add_argument('-rf', '--reward-frequency', help="How many episode before update the reward model", default=15)
parser.add_argument('-rm', '--reward-model', help="The name of the reward model", default='warrior_10')
parser.add_argument('-dn', '--dems-name', help="The name of the demonstrations file", default='dems_archer.pkl')
parser.add_argument('-fr', '--fixed-reward-model', help="Whether to use a trained reward model",
                    dest='fixed_reward_model', action='store_true')

parser.set_defaults(use_reward_model=False)
parser.set_defaults(fixed_reward_model=False)
parser.set_defaults(recurrent=False)
parser.set_defaults(parallel=False)

args = parser.parse_args()

eps = 1e-12

class OpenWorldEnv:

    def __init__(self, game_name, no_graphics, worker_id, max_episode_timesteps):
        self.no_graphics = no_graphics
        self.unity_env = UnityEnvironment(game_name, no_graphics=no_graphics, seed=worker_id, worker_id=worker_id)
        self._max_episode_timesteps = max_episode_timesteps
        self.default_brain = self.unity_env.brain_names[0]
        self.config = None
        self.actions_eps = 0.1
        self.previous_action = [0, 0]

    def execute(self, actions):
        env_info = self.unity_env.step([actions])[self.default_brain]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        self.previous_action = actions

        state = dict(global_in=env_info.vector_observations[0])
        print(actions)
        input('...')
        return state, done, reward

    def reset(self):

        self.previous_action = [0, 0]
        logs.getLogger("mlagents.envs").setLevel(logs.WARNING)
        env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
        state = dict(global_in=env_info.vector_observations[0])
        return state

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += (p * np.log(p))
        print(-entr)
        return -entr

    def set_config(self, config):
        self.config = config

    def close(self):
        self.unity_env.close()


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
        "thresholds": [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000],
        "parameters": {
            "spawn_range": [5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
            #"spawn_range": [15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            #"obstacles_already_touched": [6, 6, 5, 5, 4, 4, 3, 2, 1, 0],
            "obstacles_already_touched": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            #"obstacle_range": [9, 9, 10, 10, 11, 11, 12, 13, 14, 15],
            "obstacle_range": [15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            #"coin_range":     [15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            "coin_range":         [5, 6, 7, 8, 9, 10, 11, 12, 13, 15],
            "max_num_coin":       [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            "min_num_coin":       [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            #"max_num_coin":       [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            #"min_num_coin":       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],

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
        agent = PPO(sess, action_type='discrete', action_size=9, model_name=model_name, p_lr=7e-5,
                    v_lr=7e-5, recurrent=args.recurrent, frequency_mode=frequency_mode, distribution='gaussian',
                    p_num_itr=10, input_length=126)
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    # Open the environment with all the desired flags
    if not parallel:
        # Open the environment with all the desired flags
        env = OpenWorldEnv(game_name=None, no_graphics=True, worker_id=0, max_episode_timesteps=max_episode_timestep)
    else:
        # If parallel, create more environemnts
        envs = []
        for i in range(5):
            envs.append(OpenWorldEnv(game_name=game_name, no_graphics=True, worker_id=work_id + i, max_episode_timesteps=max_episode_timestep))

    # No IRL
    reward_model = None

    # Create runner
    if not parallel:
        runner = Runner(agent=agent, frequency=frequency, env=env, save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum,
                        frequency_mode=frequency_mode, curriculum_mode='episodes',
                        reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                        fixed_reward_model=fixed_reward_model)
    else:
        runner = ParallelRunner(agent=agent, frequency=frequency, envs=envs, save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum,
                        frequency_mode=frequency_mode, curriculum_mode='episodes',
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
