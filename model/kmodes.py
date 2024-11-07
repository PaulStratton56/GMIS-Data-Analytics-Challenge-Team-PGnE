import numpy as np
from scipy.stats import multivariate_normal


DEFAULT_CLUSTERS = 8
DEFAULT_DIM = 5

class kmodes():
    def __init__(self, n_clusters: int=DEFAULT_CLUSTERS, dim: int=DEFAULT_DIM):
        self.n_clusters = n_clusters
        self.clusters = np.random.rand(n_clusters, dim)
        self.dim = dim

    def __getkde(self, datapoints: np.array, cluster_center: np.array):
        point_sum = np.zeros((self.dim))
        for datapoint in datapoints:
            point_sum += multivariate_normal.pdf(cluster_center, mean = datapoint, cov = np.diag(np.ones(self.dim)))
        
        return (point_sum / self.n_clusters)

    def __compute_modes(self, datapoints: np.array):
        new_clusters = self.clusters
        old_clusters = 0
        iteration = 0
        while np.any(new_clusters != old_clusters):
            old_clusters = new_clusters
            new_clusters = np.zeros((self.n_clusters, self.dim))
            for index, cluster_center in enumerate(old_clusters):
                new_cluster = np.zeros((self.dim))
                for datapoint in datapoints:
                    kde = self.__getkde(datapoints, cluster_center)
                    weighted_contribution = (multivariate_normal.pdf(cluster_center, mean = datapoint, cov = np.diag(np.ones(self.dim))) * (1 / self.n_clusters) * datapoint) / kde
                    new_cluster += weighted_contribution
                new_clusters[index] = new_cluster
            print("\t\t\t\t\t\t\t\t\r", end="")
            print(f"New centers constructed. Iteration {iteration}\r", end="")
            iteration += 1
        
        return new_clusters
                
    def optimize(self, datapoints: np.array):
        # Double check to make sure that the data point dimensionality matches ours.
        assert len(datapoints.shape) == 2, "Data is not a 2D array of points (Expected dim 2, got dim " + str(len(datapoints.shape)) + ")."
        assert datapoints.shape[1] == self.dim, "Data dimensionality (" + str(datapoints.shape[1]) + ") does not match required dimensionality (" + str(self.dim) + ")." 

        # We've already initialized our centers, so it's time to compute our new modes!
        self.clusters = self.__compute_modes(datapoints)

        return self.clusters

