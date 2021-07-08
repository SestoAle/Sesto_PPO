import numpy as np
#from clustering.distance import FastDiscreteFrechetMatrix, euclidean, haversine, earth_haversine
import hdbscan
from rdp import *
import json
import matplotlib.pyplot as plt
from math import *
from world_model_3d_very import print_traj

import similaritymeasures

def compute_distance_matrix(trajectories, method="Frechet"):
    """
    :param method: "Frechet" or "Area"
    """
    n = len(trajectories)
    dist_m = np.zeros((n, n))
    for i in range(n - 1):
        p = trajectories[i]
        for j in range(i + 1, n):
            q = trajectories[j]
            if method == "Frechet":
                dist_m[i, j] = similaritymeasures.frechet_dist(p, q)
            else:
                dist_m[i, j] = similaritymeasures.area_between_two_curves(p, q)
            dist_m[j, i] = dist_m[i, j]
    return dist_m

def cluster_trajectories(trajectories):
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
    indexes = []
    for i in range(num_cluster + 1):
        index = np.where(clusterer.labels_ == i)[0][0]
        indexes.append(index)
    return indexes

if __name__ == '__main__':

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

