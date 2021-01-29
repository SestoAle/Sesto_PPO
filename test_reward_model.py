from agents.PPO import PPO
from agents.PPO_adapter import PPO as PPO_adapter
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

from reward_model.reward_model import RewardModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='warrior')
parser.add_argument('-gn', '--game-name', help="The name of the game", default='envs/DeepCrawl-Procedural-4')
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=100)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)

# Parse arguments for Inverse Reinforcement Learning
parser.add_argument('-irl', '--inverse-reinforcement-learning', dest='use_reward_model', action='store_true')
parser.add_argument('-rm', '--reward-model', help="The name of the reward model", default='warrior_10')

# Test reward models
parser.add_argument('-em', '--ensemble-mode', help="IRL", default="mult")
parser.add_argument('-t', '--temperatures', help="IRL", default="1.0,1.0,1.0")
parser.add_argument('-sn', '--save-name', help="The name for save the results", default="test_reward")

parser.set_defaults(use_reward_model=True)

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

def get_fountain_reward(state, state_n, action, env):
    
    state_n = env.get_input_observation_adapter(state_n)

    local_state = state['local_two_in']
    local_state_n = state_n['local_two_in']

    sum = local_state[:, :, 0] * 0
    for i in range(1, 8):
        sum += local_state[:, :, i] * i
    local_state = np.flip(np.transpose(sum), 0)

    sum = local_state_n[:, :, 0] * 0
    for i in range(1, 8):
        sum += local_state_n[:, :, i] * i
    local_state_n = np.flip(np.transpose(sum), 0)

    if 7 in local_state:
        if state['agent_stats'][0] < 20 and state_n['agent_stats'][0] == 20:
            return 1.0

    return 0.0

if __name__ == "__main__":

    # DeepCrawl arguments
    game_name = args.game_name
    model_name = args.model_name
    work_id = int(args.work_id)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)
    sampled_env = int(args.sampled_env)
    # IRL
    use_reward_model = args.use_reward_model
    reward_model_name = args.reward_model

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = {
        'current_step': 0,
        'thresholds': [100e6],
        'parameters':
            {
                'minTargetHp': [20],
                'maxTargetHp': [20],
                'minAgentHp': [20],
                'maxAgentHp': [20],
                'minNumLoot': [0.2],
                'maxNumLoot': [0.2],
                'minAgentMp': [0],
                'maxAgentMp': [0],
                'numActions': [17],
                # Agent statistics
                'agentAtk': [3],
                'agentDef': [3],
                'agentDes': [3],

                'minStartingInitiative': [1],
                'maxStartingInitiative': [1],

                #'sampledEnv': [sampled_env]
            }
    }

    # Total episode of training
    total_episode = 1000

    # Open the environment with all the desired flags
    env = UnityEnvWrapper(game_name, no_graphics=True, seed=int(time.time()),
                                  worker_id=work_id, with_stats=True, size_stats=31,
                                  size_global=10, agent_separate=False, with_class=False, with_hp=False,
                                  with_previous=True, verbose=False, manual_input=False,
                                  _max_episode_timesteps=max_episode_timestep)

    # Load the agents
    agents = []
    models = args.model_name.split(",")
    for m in models:
        # Create agent
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            tf.compat.v1.disable_eager_execution()
            sess = tf.compat.v1.Session(graph=graph)
            if 'fountain' in model_name:
                agent = PPO_adapter(sess=sess, model_name=model_name)
            else:
                agent = PPO(sess=sess, model_name=model_name)
            # Load agent
            agent.load_model(m, 'saved')
            agents.append(agent)

    # Create the reward models
    reward_models = []
    sessions = []

    models = args.reward_model.split(",")
    for (i, m) in enumerate(models):
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            sess = tf.compat.v1.Session(graph=graph)
            model = RewardModel(actions_size=19, policy=None, sess=sess, name='reward_model',
                                fixed_reward_model=True)
            model.load_model(m)
            print("Models loaded with name: {}".format(m))
            reward_models.append(model)

    try:
        # Evaluation loop
        current_episode = 0
        num_reward_models = len(reward_models)
        episode_rewards = dict()
        step_rewards = dict()
        all_step_rewards = dict()
        min_dict = dict()
        max_dict = dict()
        rang = 0
        for i in range(num_reward_models + 1):
            episode_rewards["reward_{}".format(i)] = []
            all_step_rewards["reward_{}".format(i)] = []
            min_dict["reward_{}".format(i)] = 99999
            max_dict["reward_{}".format(i)] = -99999

        while current_episode < total_episode:
            done = False
            current_step = 0
            for i in range(num_reward_models + 1):
                step_rewards["reward_{}".format(i)] = []
            state = env.reset(raw_obs=True)
            while not done:

                if args.ensemble_mode == 'mult':
                    total_probs = np.ones(19)
                elif args.ensemble_mode == 'add':
                    total_probs = np.zeros(19)

                main_entropy = np.inf
                min_entropy = np.inf
                min_entropy_idx = np.inf
                temperatures = [float(t) for t in args.temperatures.split(",")]
                for (i, agent) in enumerate(agents):
                    _, _, probs = agent.eval([state])
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
                    total_probs /= (num_reward_models + 1)

                if args.ensemble_mode == 'entr':
                    if min_entropy < main_entropy + 0.01:
                        #action = np.argmax(agents[min_entropy_idx].eval([state])[2])
                        action = np.argmax(agents[min_entropy_idx].eval([state])[2])
                    else:
                        #action = np.argmax(agents[0].eval([state])[2])
                        action = np.argmax(agents[0].eval([state])[2])
                elif args.ensemble_mode == 'entr_add':
                    min_entropy = np.clip(min_entropy, 0, 1)
                    main_probs = agents[0].eval([state])[2] * (min_entropy)
                    main_probs += (agents[min_entropy_idx].eval([state])[2] * (1. - min_entropy))
                    action = np.argmax(main_probs)
                else:
                    action = np.argmax(total_probs)

                state_n, done, reward = env.execute(action)

                r_fountain = get_fountain_reward(state, state_n, action)
                state = state_n

                if reward < min_dict["reward_{}".format(0)]:
                    min_dict["reward_{}".format(0)] = reward
                if reward > max_dict["reward_{}".format(0)]:
                    max_dict["reward_{}".format(0)] = reward

                step_rewards["reward_0"].append(reward)

                for (i, reward_model) in enumerate(reward_models):
                    if i == 99:
                        r = 0
                        if state['agent_stats'][6] >= 55:
                            r += 1
                        if state['agent_stats'][9] >= 79:
                            r += 1
                    elif i==99:
                        r = 0
                        if state['agent_stats'][1] == 21:
                            r += 1
                    elif i == 0:
                        r = r_fountain
                    else:
                        r = reward_model.eval([state], [state], [action])[0]
                    step_rewards["reward_{}".format(i + 1)].append(r)
                    if r < min_dict["reward_{}".format(i + 1)]:
                        min_dict["reward_{}".format(i + 1)] = r
                    if r > max_dict["reward_{}".format(i + 1)]:
                        max_dict["reward_{}".format(i + 1)] = r

                current_step += 1
                if current_step >= max_episode_timestep:
                    done = True

            for i in range(num_reward_models + 1):
                # episode_rewards["reward_{}".format(i)].append(np.sum(step_rewards["reward_{}".format(i)]))
                all_step_rewards["reward_{}".format(i)].append(step_rewards["reward_{}".format(i)])

            current_episode += 1
            print("Episode {} finished".format(current_episode))
            # for key in episode_rewards:
            #     print("{}: {}".format(key, np.mean(episode_rewards[key])))
            # print(" ")

        print("Mean of {} episode for each reward: ".format(total_episode))

        print("Saving the experiment..")
        json_str = json.dumps(all_step_rewards, cls=NumpyEncoder)
        f = open('reward_experiments/{}.json'.format(args.save_name), "w")
        f.write(json_str)
        f.close()
        print("Experiment saved with name {}!".format(args.save_name))

    finally:
        #save_model(history, model_name, curriculum, agent)
        env.close()
