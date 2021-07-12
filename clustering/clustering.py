import numpy as np
import sklearn as sk
import similaritymeasures
import hdbscan
from clustering.rdp import rdp_with_index
from joblib import Parallel, delayed
import multiprocessing

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

def thread_compute_distance(index, trajectory, trajectories):
    n = len(trajectories)
    p = trajectory
    for j in range(index + 1, n):
        q = trajectories[j]
        return similaritymeasures.frechet_dist(p, q)

def frechet_distance(p, q):
    p = np.reshape(p, (72, 3))
    q = np.reshape(q, (72, 3))
    return similaritymeasures.frechet_dist(p, q)

def parallel_compute_distance_matrix(trajectories):

    dist_m = sk.metrics.pairwise_distances(trajectories, metric=frechet_distance, n_jobs=-1)
    return dist_m


def cluster_trajectories(trajectories):

    reduced_trajectories = []
    max_length = 0
    for traj in trajectories[:500]:
        traj = np.asarray(traj)
        traj = traj[:, :3]
        new_traj, indices = rdp_with_index(traj, range(np.shape(traj)[0]), 0.01)
        if len(new_traj) > max_length:
            max_length = len(new_traj)
        reduced_trajectories.append(new_traj)

    # for traj in reduced_trajectories:
    #     for i in range(len(traj), max_length):
    #         traj.append(traj[-1])

    # reduced_trajectories = np.asarray(reduced_trajectories)
    # reduced_trajectories = np.reshape(reduced_trajectories, (-1, max_length * 3))
    print(np.shape(reduced_trajectories))
    print("alksdjadl")
    # dist_matrix = parallel_compute_distance_matrix(reduced_trajectories)

    dist_matrix = compute_distance_matrix(reduced_trajectories)
    # dist_matrix = [thread_compute_distance(i, traj, reduced_trajectories) for i, traj in enumerate(reduced_trajectories)]
    # dist_matrix = Parallel(n_jobs=num_cores)(delayed(thread_compute_distance)(i, traj, reduced_trajectories) for i, traj in enumerate(reduced_trajectories))
    # dist_matrix = parallel_compute_distance_matrix(reduced_trajectories)
    # print(np.shape(dist_matrix))
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1,
                                cluster_selection_methos='leaf')

    clusterer.fit(dist_matrix)
    num_cluster = np.max(clusterer.labels_)
    indexes = []
    for i in range(num_cluster + 1):
        index = np.where(clusterer.labels_ == i)[0][0]
        indexes.append(index)
    return indexes