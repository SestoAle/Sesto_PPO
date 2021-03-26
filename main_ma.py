from hierarchical.multi_agent import MultiAgent
from runner.ma_runner import Runner
import os
import tensorflow as tf
import argparse
import numpy as np
import math
import gym
# Load UnityEnvironment and my wrapper
from mlagents.envs import UnityEnvironment


from reward_model.reward_model import RewardModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='hierarchical')
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-sf', '--save-frequency', help="How mane episodes after save the model", default=3000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=200)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)
parser.add_argument('-rc', '--recurrent', dest='recurrent', action='store_true')

# Parse arguments for Inverse Reinforcement Learning
parser.add_argument('-irl', '--inverse-reinforcement-learning', dest='use_reward_model', action='store_true')
parser.add_argument('-rf', '--reward-frequency', help="How many episode before update the reward model", default=15)
parser.add_argument('-rm', '--reward-model', help="The name of the reward model", default='warrior_10')
parser.add_argument('-dn', '--dems-name', help="The name of the demonstrations file", default='dems_archer.pkl')
parser.add_argument('-fr', '--fixed-reward-model', help="Whether to use a trained reward model",
                    dest='fixed_reward_model', action='store_true')

# Multi agent
parser.add_argument('-cv', '--central-value', dest='central_value', action='store_true')

parser.set_defaults(use_reward_model=False)
parser.set_defaults(fixed_reward_model=False)
parser.set_defaults(recurrent=False)
parser.set_defaults(central_value=False)

args = parser.parse_args()

eps = 1e-12

class OpenWorldEnv:

    def __init__(self, game_name, no_graphics, worker_id):
        self.no_graphics = no_graphics
        self.unity_env = UnityEnvironment(game_name, no_graphics=no_graphics, seed=None, worker_id=worker_id)
        self._max_episode_timesteps = 200
        self.first_brain = self.unity_env.brain_names[0]
        self.second_brain = self.unity_env.brain_names[1]
        self.config = None
        self.actions_eps = 0.1
        self.previous_action = [0, 0]

    def execute(self, actions):

        env_info = self.unity_env.step({self.first_brain: [actions[0]], self.second_brain: [actions[1]]})

        # reward_action = 0
        # if self.previous_action is not None:
        #     if np.abs(actions[0] - self.previous_action[0]) > self.actions_eps:
        #         reward_action -= 0.2
        #     if np.abs(actions[1] - self.previous_action[1]) > self.actions_eps:
        #         reward_action -= 0.2
        #
        # self.previous_action = actions

        first_state = dict(global_in=env_info[self.first_brain].vector_observations[0])
        second_state = dict(global_in=env_info[self.second_brain].vector_observations[0])
        state = [first_state, second_state]

        done = [env_info[self.first_brain].local_done[0], env_info[self.second_brain].local_done[0]]
        reward = [env_info[self.first_brain].rewards[0], env_info[self.second_brain].rewards[0]]

        return state, done, reward

    def reset(self):
        self.previous_action = [0, 0]
        env_info = self.unity_env.reset(train_mode=True, config=self.config)

        first_state = dict(global_in=env_info[self.first_brain].vector_observations[0])
        second_state = dict(global_in=env_info[self.second_brain].vector_observations[0])
        state = [first_state, second_state]
        return state

    def entropy(self, probs):
        entr = 0
        for p in probs:
            entr += (p * np.log(p))
        return -entr

    def set_config(self, config):
        return None

    def close(self):
        self.unity_env.close()


if __name__ == "__main__":

    # DeepCrawl arguments
    model_name = args.model_name
    work_id = int(args.work_id)
    save_frequency = int(args.save_frequency)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)
    sampled_env = int(args.sampled_env)
    # IRL
    use_reward_model = args.use_reward_model
    reward_model_name = args.reward_model
    fixed_reward_model = args.fixed_reward_model
    dems_name = args.dems_name
    reward_frequency = int(args.reward_frequency)

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = None

    # Total episode of training
    total_episode = 1e10
    # Units of training (episodes or timesteps)
    frequency_mode = 'episodes'
    # Frequency of training (in episode)
    frequency = 5
    # Memory of the agent (in episode)
    memory = 10

    # Create agent
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session(graph=graph)
        agent = MultiAgent(sess=sess, num_agent=2, model_name='multi_agent', centralized_value_function=args.central_value)
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    # Open the environment with all the desired flags
    env = OpenWorldEnv(game_name=None, no_graphics=True, worker_id=work_id)

    # No IRL
    reward_model = None

    # Create runner
    runner = Runner(agents=agent, frequency=frequency, env=env, save_frequency=save_frequency,
                    logging=logging, total_episode=total_episode, curriculum=curriculum,
                    frequency_mode=frequency_mode,
                    reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                    fixed_reward_model=fixed_reward_model)

    try:
        runner.run()
    finally:
        #save_model(history, model_name, curriculum, agent)
        env.close()
