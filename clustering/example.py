import numpy as np
from clustering.distance import FastDiscreteFrechetMatrix, euclidean, haversine, earth_haversine
import hdbscan
from math import *

# Calculate distance matrix between trajectories with euclidean distance
def calculate_distance_matrix(trajectories):
    n_traj = len(trajectories)
    dist_mat = np.zeros((n_traj, n_traj), dtype=np.float64)
    dfd = FastDiscreteFrechetMatrix(earth_haversine)

    for i in range(n_traj - 1):
        p = trajectories[i]
        for j in range(i + 1, n_traj):
            q = trajectories[j]

            # Make sure the distance matrix is symmetric
            dist_mat[i, j] = dfd.distance(p, q)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat

def distance(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

def point_line_distance(point, start, end):
    if start == end:
        return distance(point, start)
    else:
        n = abs(
            (end[0] - start[0]) * (start[1] - point[1]) -
            (start[0] - point[0]) * (end[1] - start[1])
        )
        d = sqrt(
            (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2
        )
        return n / d

def rdp(points, epsilon):
    """Reduces a series of points to a simplified version that loses detail, but
    maintains the general shape of the series.
    """
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            index = i
            dmax = d

    if dmax >= epsilon:
        results = rdp(points[:index+1], epsilon)[:-1] + rdp(points[index:], epsilon)
    else:
        results = [points[0], points[-1]]

    return results

def rdp_with_index(points, indices, epsilon):
    """rdp with returned point indices
    """
    dmax, index = 0.0, 0
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], points[0], points[-1])
        if d > dmax:
            dmax, index = d, i
    if dmax >= epsilon:
        first_points, first_indices = rdp_with_index(points[:index+1], indices[:index+1], epsilon)
        second_points, second_indices = rdp_with_index(points[index:], indices[index:], epsilon)
        results = first_points[:-1] + second_points
        results_indices = first_indices[:-1] + second_indices
    else:
        results, results_indices = [points[0], points[-1]], [indices[0], indices[-1]]
    return results, results_indices

if __name__ == '__main__':

    # Create some trajectories
    trajectories = []
    for i in range(10000):
        rdp(np.random.randn(3,100), 0.01)
        input('...')
        trajectories.append(np.random.randn(3, 120))

    dist_matrix = calculate_distance_matrix(trajectories)
    print('oh')
    input('...')
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=2, min_samples=1,
                                cluster_selection_methos='leaf')

    clusterer.fit(dist_matrix)
    num_cluster = np.max(clusterer.labels_)
    print(clusterer.labels_)
    traj_to_observe = []
    for i in range(num_cluster + 1):
        print(np.where(clusterer.labels_ == i)[0][0])
        input('...')
    #print(clusterer.labels_)

