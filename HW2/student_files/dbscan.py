import numpy as np
from kmeans import pairwise_dist


class DBSCAN(object):

    def __init__(self, eps, minPts, dataset):
        self.eps = eps
        self.minPts = minPts
        self.dataset = dataset

    def regionQuery(self, pointIndex):
        """		
        Returns all points within P's eps-neighborhood (including P)
        
        Args:
            pointIndex: index of point P in dataset (self.dataset)
        Return:
            indices: (I, ) int numpy array containing the indices of all points within P's eps-neighborhood
        """
        # dist from pointIndex to all other points
        distances = pairwise_dist(self.dataset[pointIndex:pointIndex+1], self.dataset).flatten()
        
        # return all points within P's eps-neighborhood (including P)
        return np.argwhere(distances <= self.eps).flatten()

    def expandCluster(self, index, neighborIndices, C, cluster_idx, visitedIndices):
        """		
        Expands cluster C using the point P, its neighbors, and any points density-reachable to P and updates indices visited, cluster assignments accordingly
        Args:
            index: index of point P in dataset (self.dataset)
            neighborIndices: (I, ) int numpy array, indices of all points within P's eps-neighborhood
            C: current cluster as an int
            cluster_idx: (N, ) int numpy array of current assignment of clusters for each point in dataset
            visitedIndices: set of indices in dataset visited so far
        Return:
            None
        """
        # Future Devin, the comments are the lecture's pseudocode. 
        # Do not get confused by them. Sincerely, Past Devin.
        
        # assign P to cluster C
        cluster_idx[index] = C
        queue = list(neighborIndices)
        
        while queue:
            updatedP = queue.pop(0)
            if updatedP not in visitedIndices:
                # Mark P' as visited
                visitedIndices.add(updatedP)
                # NeighborPts' = regionQuery(P')
                updatedNeighborPoints = self.regionQuery(updatedP)
                # If len(NeighborPts') >= MinPts
                if len(updatedNeighborPoints) >= self.minPts:
                    # Append neighbors to queue if not already in visitedIndices or queue
                    for idx in updatedNeighborPoints:
                        if idx not in visitedIndices and idx not in queue:
                            queue.append(idx)
            # If P' is not yet member of any cluster
            if cluster_idx[updatedP] <= 0:
                # Assign P' to cluster C
                cluster_idx[updatedP] = C

    def fit(self):
        """		
        Fits DBSCAN to dataset and hyperparameters defined in init().
        Args:
            None
        Return:
            cluster_idx: (N, ) int numpy array of assignment of clusters for each point in dataset
        """
        N = self.dataset.shape[0]
        cluster_idx = np.full(N, -1, dtype=int)  # all points are unvisited; init to -1
        visitedIndices = set() # track visited
        C = 0  # init cluster count

        for index in range(N):
            if index in visitedIndices:
                continue
            visitedIndices.add(index)
            neighborIndices = self.regionQuery(index)
            if len(neighborIndices) < self.minPts:
                cluster_idx[index] = -2  # -2 to denote noise 
            else:
                # start cluster
                self.expandCluster(index, neighborIndices, C, cluster_idx, visitedIndices)
                C += 1  # increment cluster count

        # replace noise markers with -1
        cluster_idx[cluster_idx == -2] = -1

        return cluster_idx