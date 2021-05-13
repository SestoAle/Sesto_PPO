import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

name = 'bug_detector_gail_schifo_3'

with open("arrays/{}.json".format("{}_pos_buffer".format(name))) as f:
    buffer = json.load(f)

with open("arrays/{}.json".format("{}_coverage".format(name))) as f:
    coverage = json.load(f)

trajectories = None
try:
    with open("arrays/{}.json".format("{}_trajectories".format(name))) as f:
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

# Create Heatmap with matplot
data = np.zeros((40, 40))
for k in buffer.keys():
    k_value = list(map(float, k.split(" ")))
    k_value = np.asarray(k_value)
    k_value = (((k_value + 1) / 2) * 39)
    k_value = k_value.astype(int)

    # data[k_value[0], k_value[1]] += buffer[k]
    data[k_value[0], k_value[1]] = 1

data = np.clip(data, 0, np.max(data)/20)

data = np.rot90(data)

ax = plt.gca()
# Plot the heatmap
im = ax.imshow(data)

# We want to show all ticks...
# ax.set_xticks(np.arange(data.shape[1]))
# ax.set_yticks(np.arange(data.shape[0]))
# ... and label them with the respective list entries.
# ax.set_xticklabels(col_labels)
# ax.set_yticklabels(row_labels)

# # Let the horizontal axes labeling appear on top.
# ax.tick_params(top=True, bottom=False,
#                labeltop=True, labelbottom=False)
#
# # Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
#          rotation_mode="anchor")

# Turn spines off and create white grid.
for edge, spine in ax.spines.items():
    spine.set_visible(False)

ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
# ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
ax.tick_params(which="minor", bottom=False, left=False)

# Show coverage
fig = plt.figure()
plt.plot(range(len(coverage)), coverage)

def print_traj(traj):
    ep_trajectory = np.asarray(traj)
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.plot(ep_trajectory[:, 0], ep_trajectory[:, 1])

# Show last trajectory
fig = plt.figure()
threshold = 5
desired_point_x = 35
desired_point_z = 5
real_traj = None
if trajectories is not None:

    for traj in trajectories.values():

        traj = np.asarray(traj)
        for point in traj:
            if np.abs(point[0] - desired_point_x) < threshold and np.abs(point[1] - desired_point_z) < threshold:
                print_traj(traj)
                # real_traj = traj
                break

# print(real_traj)
# print_traj(real_traj)
plt.show()