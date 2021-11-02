import numpy as np
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import logging as logs
from new_unity_env_wrapper import BugEnvironment
#from mlagents.envs import UnityEnvironment

env = BugEnvironment(game_name="envs/Playtesting_3", no_graphics=True, worker_id=0,
                             max_episode_timesteps=500)
env.reset()
done = False

start = time.time()
count = 0
while count < 500:
    count += 1
    action = np.random.randint(10)
    state, done, reward = env.execute(action)
end = time.time()

print("Elapsed time: {}".format(end-start))