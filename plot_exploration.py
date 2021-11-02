import collections
import json
import os
import numpy as np
import matplotlib.pyplot as plt

all_models = ['play_4_only_im, play_4_only_im_2, play_4_only_im_3',
               'play_4_only_im_no_map, play_4_only_im_no_map_2, play_4_only_im_no_map_3']

for model_names in all_models:
    model_names = model_names.replace(" ", "")
    model_names = model_names.split(",")

    # Open the trajectories file, if exists. A trajectory is a list of points (& inventory) encountered during training
    count_buffers = []
    for model_name in model_names:
        trajectories = dict()
        for filename in os.listdir("arrays/{}/".format(model_name)):
            if 'trajectories' in filename:
                with open("arrays/{}/{}".format(model_name, filename), 'r') as f:
                    trajectories.update(json.load(f))
        trajectories = {int(k): v for k, v in trajectories.items()}
        trajectories = collections.OrderedDict(sorted(trajectories.items()))

        pos_buffer = dict()
        count_buffer = []
        count = 0
        for traj in list(trajectories.values())[:1000]:
            count += 1

            for state in traj:
                position = np.asarray(state[:3])
                position[0] = (((position[0] + 1) / 2) * 500)
                position[1] = (((position[1] + 1) / 2) * 500)
                position[2] = (((position[2] + 1) / 2) * 60)
                position = position.astype(int)
                pos_key = ' '.join(map(str, position))
                if pos_key not in pos_buffer.keys():
                    pos_buffer[pos_key] = 1

            if count % 100 == 0:
                count_buffer.append(len(list(pos_buffer.keys())))

        print("Total number of points covered by the agent: {}".format(len(list(pos_buffer.keys()))))
        print(len(count_buffer))
        count_buffers.append(count_buffer)
        # plt.plot(range(len(count_buffer)), count_buffer)
        pos_buffer = None
        trajectories = None

    means = np.mean(count_buffers, axis=0)
    stds = np.std(count_buffers, axis=0)
    plt.plot(range(len(means)), means)
    plt.fill_between(range(len(stds)), means-stds, means+stds, alpha=0.5)

plt.legend(['Our method', 'Without semantic 3D map', 'Without positional embedding'])
plt.show()
