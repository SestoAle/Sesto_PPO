from agents.PPO_openworld import PPO
from runner.runner import Runner
import os
import time
import tensorflow as tf
from unity_env_wrapper import UnityEnvWrapper
import argparse
import numpy as np
import json
import re
from utils import NumpyEncoder
import logging as logs

from reward_model.reward_model import RewardModel
from mlagents.envs import UnityEnvironment

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='openworld_discrete,openworld_discrete_coin')
parser.add_argument('-gn', '--game-name', help="The name of the game", default=None)
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=150)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)

# Test reward models
parser.add_argument('-em', '--ensemble-mode', help="IRL", default="entr_add")
parser.add_argument('-t', '--temperatures', help="IRL", default="1.0,1.0,1.0")
parser.add_argument('-sn', '--save-name', help="The name for save the results", default="test_reward")

args = parser.parse_args()

def boltzmann(probs, temperature = 1.):
    sum = np.sum(np.power(probs, 1/temperature))
    new_probs = []
    for p in probs:
        new_probs.append(np.power(p, 1/temperature) / sum)

    return np.asarray(new_probs)

def entropy(probs):
    entr = np.sum(probs * np.log(probs + 1e-12))
    return -entr

def softmax(probs, temperature = 1.):
    sum = np.sum(np.exp(probs / temperature))
    new_probs = []
    for p in probs:
        new_probs.append(np.exp(p / temperature) / sum)

    return np.asarray(new_probs)

def norm(arrays):
    return (arrays - np.min(arrays)) / (np.max(arrays) - np.min(arrays))

# Openworld environment
class OpenWorldEnv:

    def __init__(self, game_name, no_graphics, worker_id):
        self.no_graphics = no_graphics
        self.unity_env = UnityEnvironment(game_name, no_graphics=no_graphics, seed=worker_id, worker_id=worker_id)
        self._max_episode_timesteps = 150
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
        return state, done, reward

    def reset(self):
        self.previous_action = [0, 0]
        logs.getLogger("mlagents.envs").setLevel(logs.WARNING)
        env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
        state = dict(global_in=env_info.vector_observations[0])
        return state

    # Transform the observation to match the Agent
    def transform_state(self, state, long_input):
        # if long_input:
        #     return state
        # else:
        #     state = state['global_in']
        #     new_state = state[:7]
        #     for p in range(7, 71, 4):
        #         new_state = np.append(new_state, state[p])
        #
        #     return dict(global_in=new_state)
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

if __name__ == "__main__":

    # DeepCrawl arguments
    game_name = args.game_name
    model_name = args.model_name
    work_id = int(args.work_id)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)
    sampled_env = int(args.sampled_env)

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = None

    # Total episode of training
    total_episode = 10

    # Open the environment with all the desired flags
    env = OpenWorldEnv(game_name=None, no_graphics=True, worker_id=0)

    # Load the agents
    agents = []
    models = args.model_name.split(",")
    for i, m in enumerate(models):
        # Create agent
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            tf.compat.v1.disable_eager_execution()
            sess = tf.compat.v1.Session(graph=graph)
            if i == 0:
                agent = PPO(sess, action_type='discrete', action_size=9, model_name='openworld_discrete',
                            p_lr=1e-4, v_lr=1e-4, recurrent=False, frequency_mode='episodes',
                            distribution='gaussian', p_num_itr=10, input_length=70)
            else:
                agent = PPO(sess, action_type='discrete', action_size=9, model_name='openworld_discrete_obs',
                            p_lr=1e-4, v_lr=1e-4, recurrent=False, frequency_mode='episodes',
                            distribution='gaussian', p_num_itr=10, input_length=70)
            # Load agent
            agent.load_model(m, 'saved')
            agents.append(agent)

    try:
        # Evaluation loop
        current_episode = 0
        episode_rewards = dict()
        step_rewards = dict()
        all_step_rewards = dict()
        min_dict = dict()
        max_dict = dict()
        rang = 0
        all_total_rewards = []
        all_entropies = []

        while current_episode < total_episode:
            done = False
            current_step = 0
            state = env.reset()
            action = 0
            total_reward = 0
            while not done:

                if args.ensemble_mode == 'mult':
                    total_probs = np.ones(9)
                elif args.ensemble_mode == 'add':
                    total_probs = np.zeros(9)

                main_entropy = np.inf
                min_entropy = np.inf
                min_entropy_idx = np.inf
                temperatures = [float(t) for t in args.temperatures.split(",")]
                for (i, agent) in enumerate(agents):
                    if i == 0:
                        real_state = env.transform_state(state, False)
                    else:
                        real_state = env.transform_state(state, True)

                    _, _, probs = agent.eval([real_state])
                    probs = probs[0]
                    probs = boltzmann(probs, temperatures[i])
                    if args.ensemble_mode == 'mult':
                        total_probs *= probs
                    elif args.ensemble_mode == 'add':
                        total_probs += probs
                    elif args.ensemble_mode == 'entr':
                        if i == 0:
                            main_entropy = entropy(probs)
                            continue
                        if entropy(probs) < min_entropy:
                            min_entropy = entropy(probs)
                            min_entropy_idx = i
                    elif args.ensemble_mode == 'entr_add':
                        if i == 0:
                            main_entropy = entropy(probs)
                            continue
                        if entropy(probs) < min_entropy:
                            min_entropy = entropy(probs)
                            min_entropy_idx = i
                if args.ensemble_mode == 'add':
                    total_probs /= (len(agents))

                if args.ensemble_mode == 'entr':
                    if min_entropy < main_entropy + 0.01:
                        action = np.argmax(agents[min_entropy_idx].eval([state])[2])
                        #action = np.argmax(agents[min_entropy_idx].eval([env.get_input_observation_adapter(state)])[2])
                    else:
                        action = np.argmax(agents[0].eval([state])[2])
                        #action = np.argmax(agents[0].eval([env.get_input_observation(state)])[2])
                elif args.ensemble_mode == 'entr_add':
                    # Standardize min_entropy

                    #min_entropy = 1
                    all_entropies.append(min_entropy)
                    print(min_entropy)

                    #min_entropy = (min_entropy - 1.2) / (0.9)
                    min_entropy = np.clip(min_entropy, 0, 1)
                    # Transform the main state
                    main_state = env.transform_state(state, False)
                    # Transform the sub state
                    sub_state = env.transform_state(state, True)
                    main_probs = agents[0].eval([main_state])[2]
                    main_probs = boltzmann(main_probs[0], 1.0)
                    main_probs = main_probs * min_entropy

                    sub_probs = agents[min_entropy_idx].eval([sub_state])[2]
                    sub_probs = boltzmann(sub_probs[0], 1.0)
                    sub_probs *= (1 - min_entropy)

                    main_probs += sub_probs
                    main_probs = boltzmann(main_probs, 1.0)
                    #action = np.argmax(main_probs)
                    action = np.argmax(np.random.multinomial(1, main_probs))
                else:
                    action = np.argmax(total_probs)

                state_n, done, reward = env.execute(action)
                state = state_n

                total_reward += reward

                current_step += 1
                if current_step >= max_episode_timestep:
                    done = True

                if done:
                    all_total_rewards.append(total_reward)

            current_episode += 1
            print("Episode {} finished".format(current_episode))
            # for key in episode_rewards:
            #     print("{}: {}".format(key, np.mean(episode_rewards[key])))
            # print(" ")

        print("Mean of {} episode for total reward: {}".format(total_episode, np.mean(total_reward)))
        print(len(all_entropies))
        print(np.min(all_entropies))
        print(np.max(all_entropies))
        print(np.mean(all_entropies))
        print(np.std(all_entropies))
        # print("Saving the experiment..")
        # json_str = json.dumps(all_step_rewards, cls=NumpyEncoder)
        # f = open('reward_experiments/{}.json'.format(args.save_name), "w")
        # f.write(json_str)
        # f.close()
        # print("Experiment saved with name {}!".format(args.save_name))

    finally:
        #save_model(history, model_name, curriculum, agent)
        env.close()
