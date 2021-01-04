from Agents.PPO_deepcrawl import PPO
import tensorflow as tf
from unity_env_wrapper import UnityEnvWrapper
import time
import os
import json
from utils import NumpyEncoder

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Update curriculum for DeepCrawl
def set_curriculum(curriculum, total_timesteps, mode='steps'):

    global current_curriculum_step

    if curriculum == None:
        return None

    if mode == 'steps':
        lessons = np.cumsum(curriculum['thresholds'])

        curriculum_step = 0

        for (index, l) in enumerate(lessons):
            if total_timesteps > l:
                curriculum_step = index + 1


    parameters = curriculum['parameters']
    config = {}

    for (par, value) in parameters.items():
        config[par] = value[current_curriculum_step]


    current_curriculum_step = curriculum_step

    return config


def save_model(rewards, model_name):


    json_str = json.dumps(history, cls=NumpyEncoder)
    f = open("arrays/" + model_name + ".json", "w")
    f.write(json_str)
    f.close()

    print('Model saved with name: {}'.format(model_name))

# Method for count time after each episode
def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time passed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))

if __name__ == "__main__":

    # DeepCrawl
    game_name = 'envs/rnd'
    work_id = 0
    model_name = 'ranger'

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = {
        'current_step': 0,
        'thresholds': [1e6, 0.8e6, 1e6, 1e6],
        'parameters':
            {
                'minTargetHp': [1, 10, 10, 10, 10],
                'maxTargetHp': [1, 10, 20, 20, 20],
                'minAgentHp': [15, 10, 5, 5, 10],
                'maxAgentHp': [20, 20, 20, 20, 20],
                'minNumLoot': [0.2, 0.2, 0.2, 0.08, 0.04],
                'maxNumLoot': [0.2, 0.2, 0.2, 0.3, 0.3],
                'minAgentMp': [0, 0, 0, 0, 0],
                'maxAgentMp': [0, 0, 0, 0, 0],
                'numActions': [17, 17, 17, 17, 17],
                # Agent statistics
                'agentAtk': [3, 3, 3, 3, 3],
                'agentDef': [3, 3, 3, 3, 3],
                'agentDes': [3, 3, 3, 3, 3],

                'minStartingInitiative': [1, 1, 1, 1, 1],
                'maxStartingInitiative': [1, 1, 1, 1, 1]
            }
    }

    # History to save model statistics
    history = {
        "episode_rewards": [],
        "episode_timesteps": [],
        "mean_entropies": [],
        "std_entropies": [],
        "reward_model_loss": [],
        "env_rewards": []
    }

    # Total episode of training
    total_episode = 1e10
    # Frequency of training (in episode)
    frequency = 5
    # Frequency of logging
    logging = 1
    # Frequency of saving
    save_frequency = 1
    # Max timestep for episode
    max_episode_timestep = 100

    # Open the environment with all the desired flags
    env = UnityEnvWrapper(game_name, no_graphics=True, seed=int(time.time()),
                                  worker_id=work_id, with_stats=True, size_stats=31,
                                  size_global=10, agent_separate=False, with_class=False, with_hp=False,
                                  with_previous=True, verbose=False, manual_input=False,
                                  _max_episode_timesteps=max_episode_timestep)

    # Create agent
    tf.compat.v1.disable_eager_execution()
    sess = tf.compat.v1.Session()
    agent = PPO(sess=sess)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    # Training loop
    ep = 0
    total_step = 0

    # For curriculum training
    start_training = 0
    current_curriculum_step = 0

    # Trainin loop
    start_time = time.time()
    try:
        while ep <= total_episode:
            # Reset the episode
            ep += 1
            step = 0

            # Set actual curriculum
            config = set_curriculum(curriculum, total_step)
            if start_training == 0:
                print(config)
            start_training = 1
            env.set_config(config)

            state = env.reset()
            done = False
            episode_reward = 0

            # Episode loop
            while True:

                # Evaluation - Execute step
                action, logprob, probs = agent.eval([state])
                action = action[0]
                state_n, done, reward = env.execute(action)

                if step >= env._max_episode_timesteps - 1:
                    done = True

                episode_reward += reward

                # Update PPO memory
                agent.add_to_buffer(state, state_n, action, reward, logprob, done)
                state = state_n

                step += 1
                total_step += 1

                # If done, end the episode
                if done:
                    history['episode_rewards'].append(episode_reward)
                    history['episode_timesteps'].append(step)
                    history['mean_entropies'].append(episode_reward)
                    history['std_entropies'].append(episode_reward)
                    break

            # Logging information
            if ep > 0 and ep % logging == 0:
                print('Mean of {} episode reward after {} episodes: {}'.
                      format(logging, ep, np.mean(history['episode_rewards'][-logging:])))

                print('The agent made a total of {} steps'.format(total_step))

                timer(start_time, time.time())

            if ep > 0 and ep % save_frequency == 0:
                save_model(history, model_name)

            # If frequency episodes are passed, update the policy
            if ep > 0 and ep % frequency == 0:
                total_loss = agent.train()
    finally:
        save_model(history, model_name)
        env.close()