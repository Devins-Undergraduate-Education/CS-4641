from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def euclid_pairwise_dist(x: np.ndarray, y: np.ndarray) ->np.ndarray:
    """
    You implemented this in project 2! We'll give it to you here to save you the copypaste.
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
            dist: N x M array, where dist2[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
    """
    x_norm = np.sum(x ** 2, axis=1, keepdims=True)
    yt = y.T
    y_norm = np.sum(yt ** 2, axis=0, keepdims=True)
    dist2 = np.abs(x_norm + y_norm - 2.0 * (x @ yt))
    return np.sqrt(dist2)


def confusion_matrix_vis(conf_matrix: np.ndarray):
    """
    Fancy print of confusion matrix. Just encapsulating some code out of the notebook.
    """
    _, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix)
    ax.set_xlabel('Predicted Labels', fontsize=16)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Actual Labels', fontsize=16)
    for (i, j), val in np.ndenumerate(conf_matrix):
        ax.text(j, i, str(val), ha='center', va='center', bbox=dict(
            boxstyle='round', facecolor='white', edgecolor='0.3'))
    plt.show()
    return


class SMOTE(object):

    def __init__(self):
        pass

    @staticmethod
    def generate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray
        ) ->np.ndarray:
        """		
		Generate the confusion matrix for the predicted labels of a classification task.
		This function should be able to process any number of unique labels, not just a binary task.
		
		The choice to put "true" and "predicted" on the left and top respectively is arbitrary.
		In other sources, you may see this format transposed.
		
		Args:
		    y_true: (N,) array of true integer labels for the training points
		    y_pred: (N,) array of predicted integer labels for the training points
		    These vectors correspond along axis 0. y_pred[i] is the prediction for point i, whose true label is y_true[i].
		    You can assume that the labels will be ints of the form [0, u).
		Return:
		    conf_matrix: (u, u) array of ints containing instance counts, where u is the number of unique labels present
		        conf_matrix[i,j] = the number of instances where a sample from the true class i was predicted to be in class j
		"""
        raise NotImplementedError

    @staticmethod
    def f1_scores(conf_matrix: np.ndarray) ->np.ndarray:
        """		
		Given a confusion matrix, calculate F1 scores (or F-Measure) of each class.
		This function should be able to process any number of unique labels, not just a binary task.
		
		To calculate the F1 score of a given class, take the harmonic mean of the precision and recall for that class's predictions.
		Recall is the accuracy of the model on only the points of a given class.
		Precision is the proportion of all points predicted to be in a given class that are actually in the class.
		
		Note:
		    It is possible that the precision or recall are undefined (0/0) or that both of them are 0, in which case the harmonic mean is undefined.
		    In any such case, you should set the F1 score to be 0.0, as though the model had the absolute worst performance on that class.
		Args:
		    conf_matrix: (u, u) array of ints containing instance counts, where u is the number of unique class labels present
		Return:
		    f1_scores: (u,) a list containing the F1 scores for each class
		"""
        raise NotImplementedError

    @staticmethod
    def interpolate(start: np.ndarray, end: np.ndarray, inter_coeff: float
        ) ->np.ndarray:
        """		
		Return an interpolated point along the line segment between start and end.
		
		Hint:
		    if inter_coeff==0.0, this should return start;
		    if inter_coeff==1.0, this should return end;
		    if inter_coeff==0.5, this should return the midpoint between them;
		    to generalize this behavior, try writing this out in terms of vector addition and subtraction
		Args:
		    start: (D,) float array containing the start point
		    end: (D,) float array containing the end point
		    inter_coeff: (float) in [0,1] determining how far along the line segment the synthetic point should lie
		Return:
		    interpolated: (D,) float array containing the new synthetic point along the line segment
		"""
        raise NotImplementedError

    @staticmethod
    def k_nearest_neighbors(points: np.ndarray, k: int) ->np.ndarray:
        """		
		For each point, retrieve the indices of the k other points which are closest to that point.
		
		Hints:
		    Find the pairwise distances using the provided function: euclid_pairwise_dist.
		    For execution time, try to avoid looping over N, and use numpy vectorization to sort through the distances and find the relevant indices.
		Args:
		    points: (N, D) float array of points
		    k: (int) describing the number of neighbor indices to return
		Return:
		    neighborhoods: (N, k) int array containing the indices of the nearest neighbors for each point
		        neighborhoods[i, :] should be a k long 1darray containing the neighborhood of points[i]
		        neighborhoods[i, 0] = j, such that points[j] is the closest point to points[i]
		"""
        raise NotImplementedError

    @staticmethod
    def smote(X: np.ndarray, y: np.ndarray, k: int, inter_coeff_range:
        Tuple[float]) ->np.ndarray:
        """		
		Perform SMOTE on the binary classification problem (X, y), generating synthetic minority points from the minority set.
		In 6.1, we did work for an arbitrary number of classes. Here, you can assume that our problem is binary, that y will only contain 0 or 1.
		
		Outline:
		    # 1. Determine how many synthetic points are needed from which class label.
		    # 2. Get the subset of the minority points.
		    # 3. For each minority point, determine its neighborhoods. (call k_nearest_neighbors)
		    # 4. Generate |maj|-|min| synthetic data points from that subset.
		        # a. uniformly pick a random point as the start point
		        # b. uniformly pick a random neighbor as the endpoint
		        # c. uniformly pick a random interpolation coefficient from the provided range: `inter_coeff_range`
		        # d. interpolate and add to output (call interpolate)
		    # 5. Generate the class labels for these new points.
		Args:
		    X: (|maj|+|min|, D) float array of points, containing both majority and minority points; corresponds index-wise to y
		    y: (|maj|+|min|,) int array of class labels, such that y[i] is the class of X[i, :]
		    k: (int) determines the size of the neighborhood around the sampled point from which to sample the second point
		    inter_coeff_range: (a, b) determines the range from which to uniformly sample the interpolation coefficient
		        Sample U[a, b)
		        You can assume that 0 <= a < b <= 1
		Return:
		    A tuple containing:
		        - synthetic_X: (|maj|-|min|, D) float array of new, synthetic points
		        - synthetic_y: (|maj|-|min|,) array of the labels of the new synthetic points
		"""
        raise NotImplementedError
