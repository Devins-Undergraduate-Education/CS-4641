import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None
        self.mean = None

    def fit(self, X: np.ndarray) ->None:
        """		
		Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
		You may use the numpy.linalg.svd function
		Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
		corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V transpose
		
		Hint: np.linalg.svd by default returns the transpose of V
		      Make sure you remember to first center your data by subtracting the mean of each feature.
		
		Args:
		    X: (N,D) numpy array corresponding to a dataset
		
		Return:
		    None
		
		Set:
		    self.U: (N, min(N,D)) numpy array
		    self.S: (min(N,D), ) numpy array
		    self.V: (min(N,D), D) numpy array
		"""
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # perform SVD
        self.U, self.S, V_T = np.linalg.svd(X_centered, full_matrices=False)

        # V_T is the transpose of V, so we need to transpose it to get V
        self.V = V_T

    def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that final data (X_new) has K features (columns)
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    K: int value for number of columns to be kept
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        # center data
        data_centered = data - self.mean

        # prokect data onto the first K principal components
        V_K = self.V[:K, :]  # Shape: (K, D)
        X_new = np.dot(data_centered, V_K.T)  # Shape: (N, K)

        return X_new

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
        ) ->np.ndarray:
        """		
		Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
		in X_new with K features
		Utilize self.U, self.S and self.V that were set in fit() method.
		
		Args:
		    data: (N,D) numpy array corresponding to a dataset
		    retained_variance: float value for amount of variance to be retained
		
		Return:
		    X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
		           to be kept to ensure retained variance value is retained_variance
		
		Hint: Make sure you remember to first center your data by subtracting the mean of each feature.
		"""
        # calculate the cumulative variance
        total_variance = np.sum(self.S ** 2)
        cumulative_variance = np.cumsum(self.S ** 2)
        variance_ratio = cumulative_variance / total_variance

        # num of components K needed to retain the specified variance
        K = np.searchsorted(variance_ratio, retained_variance) + 1

        # transform() with the computed K
        X_new = self.transform(data, K=K)

        return X_new

    def get_V(self) ->np.ndarray:
        """		
		Getter function for value of V
		"""
        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig_title) -> None:
        """
        You have to plot three different scatterplots (2D and 3D for strongest two features and 2D for two random features) for this function.
        For plotting the 2D scatterplots, use your PCA implementation to reduce the dataset to only 2 (strongest and later random) features.
        You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color using plotly.
        Hint: Refer to https://plotly.com/python/line-and-scatter/ for making scatter plots with plotly.
        Hint: We recommend converting the data into a pandas dataframe before plotting it. Refer to https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html for more details.

        Args:
            X: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            y: (N,) numpy array, the true labels

        Return: None
        """
        self.fit(X)

        # transform X to 2D using the first two principal components
        X_pca_2D = self.transform(X, K=2)

        # create a DataFrame for 2D PCA plot
        df_pca_2D = pd.DataFrame({
            'PC1': X_pca_2D[:, 0],
            'PC2': X_pca_2D[:, 1],
            'label': y
        })

        # plot 2D scatter plot
        fig_pca_2D = px.scatter(df_pca_2D, x='PC1', y='PC2', color='label',
                                title=fig_title + ' - PCA 2D Scatter Plot')
        fig_pca_2D.update_layout(width=600, height=600)  # Set figure size
        fig_pca_2D.show()

        # transform X to 3D using the first three principal components
        X_pca_3D = self.transform(X, K=3)

        # create a DataFrame for 3D PCA plot
        df_pca_3D = pd.DataFrame({
            'PC1': X_pca_3D[:, 0],
            'PC2': X_pca_3D[:, 1],
            'PC3': X_pca_3D[:, 2],
            'label': y
        })

        # plot 3D scatter plot
        fig_pca_3D = px.scatter_3d(df_pca_3D, x='PC1', y='PC2', z='PC3', color='label',
                                title=fig_title + ' - PCA 3D Scatter Plot')
        fig_pca_3D.update_layout(width=600, height=600)  # Set figure size
        fig_pca_3D.show()

        # randomly select two features from the original data
        np.random.seed(0)  # GIVE SAME VALUES
        random_indices = np.random.choice(X.shape[1], size=2, replace=False)
        X_random_2D = X[:, random_indices]

        # create a DataFrame for random features plot
        df_random_2D = pd.DataFrame({
            'Feature1': X_random_2D[:, 0],
            'Feature2': X_random_2D[:, 1],
            'label': y
        })

        # plot 2D scatter plot of random features
        fig_random_2D = px.scatter(df_random_2D, x='Feature1', y='Feature2', color='label',
                                title=fig_title + ' - Random Features 2D Scatter Plot')
        fig_random_2D.update_layout(width=600, height=600)  # set figure size
        fig_random_2D.show()