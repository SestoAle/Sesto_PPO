from hierarchical.hierarchical_gridworld import HierarchicalAgent
from runner.hierarchical_runner import HRunner
import os
import tensorflow as tf
import argparse
from unity_env_wrapper import UnityEnvWrapper
import time


from reward_model.reward_model import RewardModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='model')
parser.add_argument('-gn', '--game-name', help="The name of the game", default=None)
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-sf', '--save-frequency', help="How many episodes after save the model", default=3000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=100)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)
parser.add_argument('-rc', '--recurrent', dest='recurrent', action='store_true')

# Parse argument for adversarial-play
parser.add_argument('-ad', '--adversarial-play', help="Whether to use adversarial play",
                    dest='adversarial_play', action='store_true')
parser.set_defaults(adversarial_play=False)

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

args = parser.parse_args()

eps = 1e-12

if __name__ == "__main__":

    # DeepCrawl arguments
    game_name = args.game_name
    model_name = args.model_name
    work_id = int(args.work_id)
    save_frequency = int(args.save_frequency)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)
    sampled_env = int(args.sampled_env)
    # Adversarial play
    adversarial_play = args.adversarial_play
    # IRL
    use_reward_model = args.use_reward_model
    reward_model_name = args.reward_model
    fixed_reward_model = args.fixed_reward_model
    dems_name = args.dems_name
    reward_frequency = int(args.reward_frequency)

    # Curriculum structure; here you can specify also the agent statistics (ATK, DES, DEF and HP)
    curriculum = {
        'current_step': 0,
        'thresholds': [100000e6, 1e6, 1e6, 1e6, 3e6, 3e6],
        'parameters':
            {
                'minTargetHp': [10],
                'maxTargetHp': [20],
                'minAgentHp': [10],
                'maxAgentHp': [20],
                'minNumLoot': [0.04],
                'maxNumLoot': [0.3],
                'minAgentMp': [0],
                'maxAgentMp': [0],
                'numActions': [17],
                # Agent statistics
                'agentAtk': [3],
                'agentDef': [3],
                'agentDes': [3],

                'minStartingInitiative': [1],
                'maxStartingInitiative': [1],

                # 'sampledEnv': [sampled_env]
            }
    }

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
        agent = HierarchicalAgent(sess=sess, manager_lr=5e-6, workers_lr=5e-6, num_workers=3,
                                  workers_name=['ranger', 'irl_buff_attack', 'irl_defence_rl'])
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    # Open the environment with all the desired flags
    env = UnityEnvWrapper(game_name, no_graphics=True, seed=int(time.time()),
                                  worker_id=work_id, with_stats=True, size_stats=31,
                                  size_global=10, agent_separate=False, with_class=False, with_hp=False,
                                  with_previous=True, verbose=False, manual_input=False,
                                  _max_episode_timesteps=max_episode_timestep,
                          )

    # No IRL
    reward_model = None

    # Create runner
    runner = HRunner(agent=agent, frequency=frequency, env=env, save_frequency=save_frequency,
                     logging=logging, total_episode=total_episode, curriculum=curriculum,
                     frequency_mode=frequency_mode,
                     reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                     fixed_reward_model=fixed_reward_model)

    try:
        runner.run()
    finally:
        #save_model(history, model_name, curriculum, agent)
        env.close()
