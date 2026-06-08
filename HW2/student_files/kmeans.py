"""
File: kmeans.py
Project: Downloads
File Created: Feb 2021
Author: Rohit Das
"""
import numpy as np
np.random.seed(37)


class KMeans(object):

    def __init__(self, points, k, init='random', max_iters=10000, rel_tol=1e-05
        ):
        """
        Args:
            points: NxD numpy array, where N is # points and D is the dimensionality
            K: number of clusters
            init : how to initial the centers
            max_iters: maximum number of iterations (Hint: You could change it when debugging)
            rel_tol: convergence criteria with respect to relative change of loss (number between 0 and 1)
        Return:
            none
        """
        self.points = points
        self.K = k
        if init == 'random':
            self.centers = self.init_centers()
        else:
            self.centers = self.kmpp_init()
        self.assignments = None
        self.loss = 0.0
        self.rel_tol = rel_tol
        self.max_iters = max_iters

    def init_centers(self):
        """		
            Initialize the centers randomly
        Return:
            self.centers : K x D numpy array, the centers.
        Hint: Please initialize centers by randomly sampling points from the dataset in case the autograder fails.
        """
        uniquePoints = np.unique(self.points, axis=0)  # get unique points
        numUniquePoints = uniquePoints.shape[0]  # count unique points
        # randomly select centers from the unique points
        return uniquePoints[np.random.permutation(numUniquePoints)[:min(self.K, numUniquePoints)]]

    def kmpp_init(self):
        """		
            Use the intuition that points further away from each other will probably be better initial centers.
            To complete this method, refer to the steps outlined below:.
            1. Sample 1% of the points from dataset, uniformly at random (UAR) and without replacement.
            This sample will be the dataset the remainder of the algorithm uses to minimize initialization overhead.
            2. From the above sample, select only one random point to be the first cluster center.
            3. For each point in the sampled dataset, find the nearest cluster center and record the squared distance to get there.
            4. Examine all the squared distances and take the point with the maximum squared distance as a new cluster center.
            In other words, we will choose the next center based on the maximum of the minimum calculated distance
            instead of sampling randomly like in step 2. You may break ties arbitrarily.
            5. Repeat 3-4 until all k-centers have been assigned. You may use a loop over K to keep track of the data in each cluster.
        Return:
            self.centers : K x D numpy array, the centers.
        Hint:
            You could use functions like np.vstack() here.
        """
        N, D = self.points.shape  # set vars
        N_sample = max(1, int(0.01 * N))  # ensure min 1 sample
        sampleIndexes = np.random.choice(N, N_sample, replace=False)
        samplePts = self.points[sampleIndexes, :] 

        # step Dos
        firstCenterIndex = np.random.choice(N_sample, 1, replace=False)
        centers = samplePts[firstCenterIndex, :].reshape(1, -1)  # initalize centers array

        minDist = np.sum((samplePts - centers[0]) ** 2, axis=1)

        # selet remaining centers
        for _ in range(1, self.K):
            # step fo
            max_index = np.argmax(minDist)
            new_center = samplePts[max_index, :].reshape(1, -1)
            centers = np.vstack((centers, new_center))  # add new center

            # step tree
            dist = np.sum((samplePts - new_center) ** 2, axis=1)
            minDist = np.minimum(minDist, dist)
    
        self.centers = centers # assign centers
        return self.centers

    def update_assignment(self):
        """	
            Update the membership of each point based on the closest center
        Return:
            self.assignments : numpy array of length N, the cluster assignment for each point
        Hint: Do not use loops for the update_assignment function
        Hint: You could call pairwise_dist() function
        Hint: In case the np.sqrt() function is giving an error in the pairwise_dist() function, you can use the squared distances directly for comparison.
        """
        dists = pairwise_dist(self.points, self.centers)  # N x K

        # reassign point
        self.assignments = np.argmin(dists, axis=1)  # N x ,

        return self.assignments

    def update_centers(self):
        """		
            update the cluster centers
        Return:
            self.centers: new centers, a new K x D numpy array of float dtype, where K is the number of clusters, and D is the dimension.
        
        HINT: Points may be integer, but the centers should not have to be. Watch out for dtype casting!
        HINT: If there is an empty cluster then it won't have a cluster center, in that case the number of rows in self.centers can be less than K.
        """
        N, D = self.points.shape
        new_centers = np.zeros((self.K, D), dtype=np.float64) # RAHHHHH TYPECASTING
        for k in range(self.K):
            indices = np.where(self.assignments == k)[0]
            if len(indices) > 0:
                cluster_points = self.points[indices]
                center = np.mean(cluster_points, axis=0, dtype=np.float64) # make literally everything a float
                new_centers[k] = center
            else:
                # reinitalize empty cluster center to a random point
                random_index = np.random.choice(N)
                new_centers[k] = self.points[random_index]
        self.centers = new_centers
        return self.centers



    def get_loss(self):
        """		
            The loss will be defined as the sum of the squared distances between each point and it's respective center.
        Return:
            self.loss: a single float number, which is the objective function of KMeans.
        """
        assigned_centers = self.centers[self.assignments] # N x D
        squared_dists = np.sum((self.points - assigned_centers) ** 2, axis=1) # N x ,
        self.loss = np.sum(squared_dists) # float
        
        return self.loss
    
    def train(self):
        """		
            Train KMeans to cluster the data:
                0. Recall that centers have already been initialized in __init__
                1. Update the cluster assignment for each point
                2. Update the cluster centers based on the new assignments from Step 1
                3. Check to make sure there is no mean without a cluster,
                   i.e. no cluster center without any points assigned to it.
                   - In the event of a cluster with no points assigned,
                     pick a random point in the dataset to be the new center and
                     update your cluster assignment accordingly.
                4. Calculate the loss and check if the model has converged to break the loop early.
                   - The convergence criteria is measured by whether the percentage difference
                     in loss compared to the previous iteration is less than the given
                     relative tolerance threshold (self.rel_tol).
                   - Relative tolerance threshold (self.rel_tol) is a number between 0 and 1.
                5. Iterate through steps 1 to 4 max_iters times. Avoid infinite looping!
        
        Return:
            self.centers: K x D numpy array, the centers
            self.assignments: Nx1 int numpy array
            self.loss: final loss value of the objective function of KMeans.
        
        HINT: Do not loop over all the points in every iteration. This may result in time out errors
        HINT: Make sure to care of empty clusters. If there is an empty cluster the number of rows in self.centers can be less than K.
        """
        prev_loss = None
        N, D = self.points.shape
        
        for i in range(self.max_iters):
            self.update_assignment() # step uno
            self.update_centers() # step dos
            
            # step tree
            empty_clusters = []
            for k in range(self.K):
                if not np.any(self.assignments == k): # if no pts assigned to k
                    empty_clusters.append(k)
            
            # edge case (kinda) for empty cluster
            for k in empty_clusters:
                random_index = np.random.choice(N)
                self.centers[k] = self.points[random_index]
            
            # step for
            current_loss = self.get_loss()
            
            if prev_loss is not None:
                relative_change = abs(current_loss - prev_loss) / prev_loss
                if relative_change < self.rel_tol:
                    # print(f"Converged after {iteration + 1} iterations.") # funky sanity check
                    break
            
            prev_loss = current_loss
            
        self.update_assignment()
        self.update_centers()
        
        return self.centers, self.assignments, self.loss



def pairwise_dist(x, y):
    """	
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
    
    HINT: Do not use loops for the pairwise_distance function
    """
    x_sq_sum = np.sum(np.square(x), axis=1)[:, np.newaxis]  # N x 1
    y_sq_sum = np.sum(np.square(y), axis=1)  # M x 1
    xy_dot_prod = x.dot(y.T)  # N x M

    return np.sqrt(np.clip(x_sq_sum + y_sq_sum - 2 * xy_dot_prod, 0, None))


def fowlkes_mallow(xGroundTruth, xPredicted):
    """	
    Args:
        xPredicted : list of length N where N = no. of test samples
        xGroundTruth: list of length N where N = no. of test samples
    Return:
        fowlkes-mallow value: final coefficient value as a float
    
    HINT: You can use loops for this function.
    HINT: The idea is to make the comparison of Predicted and Ground truth in pairs.
        1. Choose a pair of points from the Prediction.
        2. Compare the prediction pair pattern with the ground truth pair.
        3. Based on the analysis, we can figure out whether it's a TP/FP/FN/FP.
        4. Then calculate fowlkes-mallow value
    """

    # initalize counts
    TP = FP = FN = 0

    # create label arrays
    xGroundTruth = np.array(xGroundTruth)
    xPredicted = np.array(xPredicted)

    # loop over pairs
    for i in range(len(xGroundTruth)):
        for j in range(i + 1, len(xGroundTruth)):
            same_cluster_true = xGroundTruth[i] == xGroundTruth[j]
            same_cluster_pred = xPredicted[i] == xPredicted[j]
            if same_cluster_pred and same_cluster_true:
                TP += 1  # true pos
            elif same_cluster_pred and not same_cluster_true:
                FP += 1  # false pos
            elif not same_cluster_pred and same_cluster_true:
                FN += 1  # false neg

    # return calculations
    return float(TP / np.sqrt((TP + FP) * (TP + FN)))