from mlagents.envs import UnityEnvironment

import numpy as np

import signal
import time
import logging
import math

eps = 1e-12

class UnityEnvWrapper:

    def __init__(self, game_name=None, no_graphics=True, seed=None, worker_id=0, config=None, _max_episode_timesteps=30):

        self.game_name = game_name
        self.no_graphics = no_graphics
        self.seed = seed
        self.worker_id = worker_id
        self.unity_env = self.open_unity_environment(game_name, no_graphics, seed, worker_id)
        self.default_brain = self.unity_env.brain_names[0]
        self._max_episode_timesteps = _max_episode_timesteps

        self.set_config(config)

    def states(self):
        return dict(position=dict(type='float', shape=(2,)),
                    target_position=dict(type='float', shape=(2,)),
                    env_objects_distances=dict(type='float', shape=(5,)),
                    in_range=dict(type='float', shape=(1,)))

    def actions(self):
        # Horizontal, Vertical, Attack
        return dict(type='float', shape=(3,), min_value=-1.0, max_value=1.0)

    def reset(self):

        env_info = None

        while env_info == None:

            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(60)

            try:
                logging.getLogger("mlagents.envs").setLevel(logging.WARNING)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]

            except Exception as exc:
                self.close()
                self.unity_env = self.open_unity_environment(self.game_name, self.no_graphics, seed=int(time.time()),
                                                             worker_id=self.worker_id)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
                print("The environment didn't respond, it was necessary to close and reopen it")

        obs = self.get_input_observation(env_info)

        return obs

    def execute(self, actions):

        env_info = None

        signal.alarm(0)

        while env_info == None:

            signal.signal(signal.SIGALRM, self.handler)
            signal.alarm(3000)

            try:
                env_info = self.unity_env.step([actions])[self.default_brain]

            except Exception as exc:
                self.close()
                self.unity_env = self.open_unity_environment(self.game_name, self.no_graphics, seed=int(time.time()),
                                                             worker_id=self.worker_id)
                env_info = self.unity_env.reset(train_mode=True, config=self.config)[self.default_brain]
                print("The environment didn't respond, it was necessary to close and reopen it")

        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        observation = self.get_input_observation(env_info)


        return [observation, done, reward]

    def open_unity_environment(self, game_name, no_graphics, seed, worker_id):
        return UnityEnvironment(game_name, no_graphics=no_graphics, seed=seed, worker_id=worker_id)

    def close(self):
        self.unity_env.close()

    def set_config(self, config):
        self.config = config

    def handler(self, signum, frame):
        print("Timeout!")
        raise Exception("end of time")

    def get_input_observation(self, env_info):

        observation = {
            'position': np.asarray(env_info.vector_observations[0][:2]),
            'target_position': np.asarray(env_info.vector_observations[0][2:4]),
            'env_objects_distances': np.asarray(env_info.vector_observations[0][4:9]),
            'in_range': np.asarray(env_info.vector_observations[0][9:10])
        }
        observation = env_info.vector_observations[0]
        observation = dict(global_in=observation)

        return observation

    def entropy(self, probs):

        return 0