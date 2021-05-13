import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from motivation.random_network_distillation import RND
from architectures.bug_arch import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

name1 = "bug_detector_gail_schifo_3"
name2 = "bug_detector_gail_schifo_moti"
name3 = "bug_detector_gail_schifo_irl"

model_name = name1

with open("arrays/{}.json".format("{}_pos_buffer".format(model_name))) as f:
    buffer = json.load(f)

with open("arrays/{}.json".format("{}_coverage".format(model_name))) as f:
    coverage = json.load(f)

trajectories = None
try:
    with open("arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
        trajectories = json.load(f)
except Exception as e:
    pass

# Saving Heatmap with PIL
img = Image.new('RGB', (20, 20))
for k in buffer.keys():
    k_value = list(map(float, k.split(" ")))
    k_value = np.asarray(k_value)
    k_value = (((k_value + 1) / 2) * 19)
    k_value = k_value.astype(int)

    img.putpixel(k_value[:2], (155, 155, 155))

img.save('sqr.png')

def plot_map(map):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(map)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(heatmap.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(heatmap.shape[0] + 1) - .5, minor=True)
    # ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

# Create Heatmap with matplot
heatmap = np.zeros((40, 40))
covmap = np.zeros((40, 40))
for k in buffer.keys():
    k_value = list(map(float, k.split(" ")))
    k_value = np.asarray(k_value)
    k_value = (((k_value + 1) / 2) * 39)
    k_value = k_value.astype(int)

    heatmap[k_value[0], k_value[1]] += buffer[k]
    covmap[k_value[0], k_value[1]] = 1

heatmap = np.clip(heatmap, 0, np.max(heatmap) / 20)

heatmap = np.rot90(heatmap)
covmap = np.rot90(covmap)

# Plot heatmap
plt.figure()
plot_map(heatmap)

# Plot coverage map
plt.figure()
plot_map(covmap)

# Show coverage
fig = plt.figure()
plt.plot(range(len(coverage)), coverage)

def print_traj(traj):
    ep_trajectory = np.asarray(traj)
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.plot(ep_trajectory[:, 0], ep_trajectory[:, 1])

# # Show last trajectory
# fig = plt.figure()
# threshold = 5
# desired_point_x = 35
# desired_point_z = 5
# real_traj = None
# if trajectories is not None:
#
#     for traj in trajectories.values():
#
#         traj = np.asarray(traj)
#         for point in traj:
#             if np.abs(point[0] - desired_point_x) < threshold and np.abs(point[1] - desired_point_z) < threshold:
#                 print_traj(traj)
#                 # real_traj = traj
#                 break
#
# # print(real_traj)
# # print_traj(real_traj)


'''
    Compute the cumulative value of the learnt RND to compare trajectories
'''
if trajectories is not None:
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        tf.compat.v1.disable_eager_execution()
        motivation_sess = tf.compat.v1.Session(graph=graph)
        motivation = RND(motivation_sess, input_spec=input_spec, network_spec=network_spec_rnd,
                         obs_to_state=obs_to_state_rnd)
        init = tf.compat.v1.global_variables_initializer()
        motivation_sess.run(init)
        motivation.load_model(name=model_name, folder='saved')

    filler = np.zeros((42))
    traj_to_observe = []
    rnd_batch = []
    desired_point_x = 35
    desired_point_z = 5
    threshold = 5
    moti_rews = []
    for traj in trajectories.values():
        for point in traj:
            if np.abs(point[0] - desired_point_x) < threshold and np.abs(point[1] - desired_point_z) < threshold:
                traj_to_observe.append(traj)
                break

    for traj in traj_to_observe:
        rnd_batch = []
        for state in traj:
            state = np.asarray(state)
            state = 2 * (state/40) - 1
            state = np.concatenate([state, filler])
            state = dict(global_in=state)
            rnd_batch.append(state)

        moti_rew = np.sum(motivation.eval(rnd_batch))
        moti_rews.append(moti_rew)

    moti_mean = np.mean(moti_rews)
    idxs_to_observe = np.where(moti_rews > np.asarray(15.))
    idxs_to_observe = np.reshape(idxs_to_observe, -1)
    traj_to_observe = np.asarray(traj_to_observe)
    for traj in traj_to_observe[idxs_to_observe]:
        print_traj(traj)


plt.show()