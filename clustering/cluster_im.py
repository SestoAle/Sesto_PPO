import torch
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min
import hdbscan
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# Cluster through latent space of autoencoder.
# The autoencoder is made in Torch
# TODO: move the autoencoder to TF
def cluster(im_rews, clustering_mode='kmeans', clusters=10):

    # The Imitation rewards will be our latent space to cluster
    latents = np.asarray(im_rews)

    # Cluster through Spectral
    if clustering_mode == 'spectral':
        clusterer = SpectralClustering(clusters).fit(latents)
        closest = []
        num_cluster = np.max(clusterer.labels_)
        for i in range(num_cluster + 1):
            index = np.where(clusterer.labels_ == i)[0][0]
            closest.append(index)
    elif clustering_mode == 'kmeans':
        clusterer = KMeans(clusters).fit(latents)
        num_cluster = np.max(clusterer.labels_)
        closest, _ = pairwise_distances_argmin_min(clusterer.cluster_centers_, latents)
        closest = np.asarray(closest)
    elif clustering_mode == 'hdbscan':
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=1).fit(latents)
        closest = []
        num_cluster = np.max(clusterer.labels_)
        for i in range(num_cluster + 1):
            index = np.where(clusterer.labels_ == i)[0][0]
            closest.append(index)

    # Return the indices of the trajectories that define each cluster
    print('Clustering done! Num cluster: {}'.format(num_cluster + 1))
    return np.asarray(closest)

if __name__ == '__main__':

    trajectories = np.load('../traj_to_observe.npy')
    cluster(trajectories, 'autoencoders/labyrinth')