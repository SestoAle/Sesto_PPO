from agents.PPO_gradi import PPO
from runner.runner import Runner
from runner.parallel_runner import Runner as ParallelRunner
import os
import time
import tensorflow as tf
from env_gradi import UnityEnvWrapper
import argparse

from reward_model.reward_model import RewardModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# Parse arguments for training
parser = argparse.ArgumentParser()
parser.add_argument('-mn', '--model-name', help="The name of the model", default='model')
parser.add_argument('-gn', '--game-name', help="The name of the game", default="envs/DeepCrawl-Procedural-4")
parser.add_argument('-wk', '--work-id', help="Work id for parallel training", default=0)
parser.add_argument('-sf', '--save-frequency', help="How many episodes after save the model", default=100000)
parser.add_argument('-lg', '--logging', help="How many episodes after logging statistics", default=100)
parser.add_argument('-mt', '--max-timesteps', help="Max timestep per episode", default=30)
parser.add_argument('-se', '--sampled-env', help="IRL", default=20)
parser.add_argument('-rc', '--recurrent', dest='recurrent', action='store_true')
parser.add_argument('-pl', '--parallel', dest='parallel', action='store_true')

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
parser.set_defaults(parallel=False)

args = parser.parse_args()

if __name__ == "__main__":

    # DeepCrawl arguments
    game_name = args.game_name
    model_name = args.model_name
    work_id = int(args.work_id)
    save_frequency = int(args.save_frequency)
    logging = int(args.logging)
    max_episode_timestep = int(args.max_timesteps)
    sampled_env = int(args.sampled_env)
    # Whether to use parallel executions
    parallel = args.parallel
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
        "thresholds": [8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000, 8000],
        "parameters": {
            "range": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "agent_fixed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "target_fixed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "agent_update_rate": [15, 15, 15, 15, 15, 15, 15, 15, 15, 15],
            "speed": [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            "update_movement": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "attack_range_epsilon": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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
        agent = PPO(sess=sess, memory=memory, model_name=model_name, recurrent=args.recurrent,
                    frequency_mode=frequency_mode, action_type='continuous', action_size=3, p_lr=5e-5, v_lr=5e-4,
                    distribution='beta', action_min_value=-1, action_max_value=1)
        # Initialize variables of models
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

    double_agent = None
    # Create double agent
    if adversarial_play:
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            double_sess = tf.compat.v1.Session(graph=graph)
            double_agent = PPO(sess=double_sess, memory=memory, model_name=model_name, recurrent=args.recurrent)
            # Initialize variables of models
            init = tf.compat.v1.global_variables_initializer()
            double_sess.run(init)

    # Open the environment with all the desired flags
    if not parallel:
        env = UnityEnvWrapper(game_name, no_graphics=True, seed=int(time.time()),
                                  worker_id=work_id,
                                  _max_episode_timesteps=max_episode_timestep,
                          )
    else:
        # If parallel, create more environemnts
        envs = []
        for i in range(5):
            envs.append(UnityEnvWrapper(game_name, no_graphics=True, seed=int(time.time() + i),
                                  worker_id=work_id + i,
                          ))

    # If we want to use IRL, create a reward model
    reward_model = None
    if use_reward_model:
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            reward_sess = tf.compat.v1.Session(graph=graph)
            reward_model = RewardModel(actions_size=19, policy=agent, sess=reward_sess, name=model_name)
            # Initialize variables of models
            init = tf.compat.v1.global_variables_initializer()
            reward_sess.run(init)
            # If we want, we can use an already trained reward model
            if fixed_reward_model:
                reward_model.load_model(reward_model_name)
                print("Model loaded!")

    # Create runner
    if not parallel:
        runner = Runner(agent=agent, frequency=frequency, env=env, save_frequency=save_frequency,
                    logging=logging, total_episode=total_episode, curriculum=curriculum,
                    frequency_mode=frequency_mode,
                    reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                    fixed_reward_model=fixed_reward_model, curriculum_mode='episodes',

                    # Adversarial play
                    double_agent=double_agent, adversarial_play=adversarial_play
                    )
    else:
        runner = ParallelRunner(agent=agent, frequency=frequency, envs=envs, save_frequency=save_frequency,
                        logging=logging, total_episode=total_episode, curriculum=curriculum,
                        frequency_mode=frequency_mode,
                        reward_model=reward_model, reward_frequency=reward_frequency, dems_name=dems_name,
                        fixed_reward_model=fixed_reward_model,

                        # Adversarial play
                        double_agent=double_agent, adversarial_play=adversarial_play
                        )
    try:
        runner.run()
    finally:
        if not parallel:
            env.close()
        else:
            for env in envs:
                env.close()
