import json
import matplotlib.pyplot as plt
from clustering.clustering import *

if __name__ == '__main__':

    from world_model_3d_very import print_traj
    # Load trajectories
    trajectories = None
    model_name = 'double_jump_impossibru_both'
    with open("../arrays/{}.json".format("{}_trajectories".format(model_name))) as f:
        trajectories = json.load(f)

    trajectories_to_cluster = []
    for traj in list(trajectories.values())[-100:]:
        traj = np.asarray(traj)
        traj = traj[:, :3]
        new_traj, indices = rdp_with_index(traj, range(np.shape(traj)[0]), 0.01)
        trajectories_to_cluster.append(new_traj)

    dist_matrix = compute_distance_matrix(trajectories_to_cluster)
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1,
                                cluster_selection_methos='leaf')

    clusterer.fit(dist_matrix)
    num_cluster = np.max(clusterer.labels_)
    traj_to_observe = []
    for i in range(num_cluster + 1):
        index = np.where(clusterer.labels_ == i)[0][0]
        print_traj(list(trajectories.values())[-100:][index], 'g')
        plt.show()
        plt.waitforbuttonpress()

