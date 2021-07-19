import numpy as np
import sklearn as sk
import similaritymeasures
import hdbscan
from clustering.rdp import rdp_with_index, distance
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
    #distances = np.zeros((n, n))
    distances = dict()
    for j in range(index + 1, n):
        q = trajectories[j]
        distances['{},{}'.format(index,j)] = distances['{},{}'.format(j,index)] = similaritymeasures.frechet_dist(p, q)
        # dist_matrix[index, j] = dist_matrix[j, index] = similaritymeasures.frechet_dist(p, q)
    return distances

def my_frechet_dist(p, q):
    dists = np.zeros(len(p))
    for i, pp, pq in zip(range(len(p), p, q)):
        dists[i] = distance(pp, pq)
    return np.abs(np.min(p - q))


def cluster_trajectories(trajectories):

    all_reduced_trajectories = []
    max_length = 0
    mean_length = 0
    for traj in trajectories:
        traj = np.asarray(traj)
        traj = traj[:, :3]
        new_traj, indices = rdp_with_index(traj, range(np.shape(traj)[0]), 0.5)
        if len(new_traj) > max_length:
            max_length = len(new_traj)
        mean_length += len(new_traj)
        all_reduced_trajectories.append(new_traj)

    print("mean_length: {}".format(mean_length/len(trajectories)))

    num_chunk = 1
    chunk_size = int(len(trajectories) / num_chunk)
    indexes = []
    for j in range(num_chunk):
        reduced_trajectories = all_reduced_trajectories[j*chunk_size:j*chunk_size + chunk_size]
        num_cores = multiprocessing.cpu_count()
        import time
        start = time.time()
        dist_matrix = np.zeros((len(reduced_trajectories), len(reduced_trajectories)))
        dist_matrices = Parallel(n_jobs=num_cores)(
            delayed(thread_compute_distance)(i, traj, reduced_trajectories)
           for i, traj in enumerate(reduced_trajectories))

        for dist in dist_matrices:
            for key in dist.keys():
                indeces = [int(k) for k in key.split(',')]
                dist_matrix[indeces[0], indeces[1]] += dist[key]
        end = time.time()
        print(end - start)

        # dist_matrix = compute_distance_matrix(reduced_trajectories)
        # dist_matrix = parallel_compute_distance_matrix(reduced_trajectories)
        clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=15, min_samples=1,
                                    cluster_extraction_method="leaf")

        clusterer.fit(dist_matrix)
        num_cluster = np.max(clusterer.labels_)
        for i in range(num_cluster + 1):
            index = np.where(clusterer.labels_ == i)[0][0]
            indexes.append(j * chunk_size + index)
        # try:
        #     index = np.where(clusterer.labels_ == -1)[0][0]
        #     indexes.append(j * chunk_size + index)
        # except Exception as e:
        #     pass
    return indexes