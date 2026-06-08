from typing import List, Tuple
import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) ->float:
        """		
        Calculate the root mean square error.
        
        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        return np.sqrt(np.mean((pred - label) ** 2))

    def construct_polynomial_feats(self, x: np.ndarray, degree: int
        ) ->np.ndarray:
        """		
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree
        
        Args:
            x:
                1-dimensional case: (N,) numpy array
                D-dimensional case: (N, D) numpy array
                Here, N is the number of instances and D is the dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        Hints:
            - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
            the bias term.
            - It is acceptable to loop over the degrees.
            - Example:
            For inputs x: (N = 3 x D = 2) and degree: 3,
            feat should be:
        
            [[[ 1.0        1.0]
                [ x_{1,1}    x_{1,2}]
                [ x_{1,1}^2  x_{1,2}^2]
                [ x_{1,1}^3  x_{1,2}^3]]
        
                [[ 1.0        1.0]
                [ x_{2,1}    x_{2,2}]
                [ x_{2,1}^2  x_{2,2}^2]
                [ x_{2,1}^3  x_{2,2}^3]]
        
                [[ 1.0        1.0]
                [ x_{3,1}    x_{3,2}]
                [ x_{3,1}^2  x_{3,2}^2]
                [ x_{3,1}^3  x_{3,2}^3]]]
        """
        if x.ndim == 1:
            x = x[:, np.newaxis] # (N, 1)
            reshape_output = True
        else:
            reshape_output = False

        N, D = x.shape
        degrees = np.arange(degree + 1)  # (degree+1,)

        # expand dims
        x_expanded = x[:, np.newaxis, :] # (N, 1, D)
        degrees_expanded = degrees[np.newaxis, :, np.newaxis] # (1, degree+1, 1)

        # calculate x raised to each degree
        feat = x_expanded ** degrees_expanded # (N, degree+1, D)

        if reshape_output:
            # (N, degree+1)
            feat = feat.reshape(N, degree + 1)

        return feat

    def predict(self, xtest: np.ndarray, weight: np.ndarray) ->np.ndarray:
        """		
        Using regression weights, predict the values for each data point in the xtest array
        
        Args:
            xtest: (N,1+D) numpy array, where N is the number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            weight: (1+D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        return xtest @ weight

    def linear_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray
        ) ->np.ndarray:
        """		
        Fit a linear regression model using the closed form solution
        
        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you should use the numpy linear algebra function (np.linalg.pinv)
        """
        X_pinv = np.linalg.pinv(xtrain)  # (D+1, N)
        return X_pinv @ ytrain

    def linear_fit_GD(self, xtrain: np.ndarray, ytrain: np.ndarray, epochs:
        int=5, learning_rate: float=0.001) ->Tuple[np.ndarray, List[float]]:
        """		
        Fit a linear regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.
        
        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
        """
        N, D_plus1 = xtrain.shape
        # init weights to zeros
        weight = np.zeros((D_plus1, 1)) # (D+1, 1)
        loss_per_epoch = []

        for epoch in range(epochs):
            # compute predictions
            y_pred = xtrain @ weight # (N, 1)
            # compute error
            error = y_pred - ytrain # (N, 1)
            # compute gradient
            gradient = (1 / N) * (xtrain.T @ error) # (D+1, 1)
            # update weights
            weight = weight - learning_rate * gradient
            # compute RMSE after weight update
            y_pred_updated = xtrain @ weight
            rmse = self.rmse(y_pred_updated, ytrain)
            loss_per_epoch.append(rmse)

        return weight, loss_per_epoch

    def linear_fit_SGD(self, xtrain: np.ndarray, ytrain: np.ndarray, epochs:
        int=100, learning_rate: float=0.001) ->Tuple[np.ndarray, List[float]]:
        """		
        Fit a linear regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.
        
        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
            - Keep in mind that the number of epochs is the number of complete passes
            through the training dataset. SGD updates the weight for one datapoint at
            a time. For each epoch, you'll need to go through all of the points.
        
        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
        """
        N, D_plus1 = xtrain.shape
        # init weights to zeros
        weight = np.zeros((D_plus1, 1)) # (D+1, 1)
        loss_per_step = []

        for epoch in range(epochs):
            for i in range(N):
                xi = xtrain[i:i+1, :] # (1, D+1)
                yi = ytrain[i:i+1, :] # (1, 1)
                # compute prediction for xi
                y_pred_i = xi @ weight # (1, 1)
                # compute error
                error = y_pred_i - yi # (1, 1)
                # compute gradient
                gradient = xi.T @ error # (D+1, 1)
                # update weights
                weight = weight - learning_rate * gradient
                # compute RMSE over the entire training set
                y_pred_full = xtrain @ weight # (N, 1)
                rmse = self.rmse(y_pred_full, ytrain)
                loss_per_step.append(rmse)

        return weight, loss_per_step

    def ridge_fit_closed(self, xtrain: np.ndarray, ytrain: np.ndarray,
        c_lambda: float) ->np.ndarray:
        """		
        Fit a ridge regression model using the closed form solution
        
        Args:
            xtrain: (N,1+D) numpy array, where N is
                    number of instances and D is the dimensionality
                    of each instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
        Return:
            weight: (1+D,1) numpy array, the weights of ridge regression model
        Hints:
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        N, D_plus1 = xtrain.shape
        # compute X^T X
        XTX = xtrain.T @ xtrain  # (D+1, D+1)
        # create regularization matrix
        L = np.eye(D_plus1)
        L[0, 0] = 0  # NO REGULARIZE
        # compute the regularized matrix
        XTX_reg = XTX + c_lambda * L
        # compute X^T y
        XTy = xtrain.T @ ytrain  # (D+1, 1)
        # solve for weights
        weight = np.linalg.inv(XTX_reg) @ XTy
        return weight

    def ridge_fit_GD(self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda:
        float, epochs: int=500, learning_rate: float=1e-07) ->Tuple[np.
        ndarray, List[float]]:
        """		
        Fit a ridge regression model using gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.
        
        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
            - You should avoid applying regularization to the bias term in the gradient update
        """
        N, D_plus1 = xtrain.shape
        weight = np.zeros((D_plus1, 1))
        loss_per_epoch = []

        for epoch in range(epochs):
            y_pred = xtrain @ weight
            error = y_pred - ytrain
            # regularization term
            regularization = c_lambda * weight
            regularization[0, 0] = 0  # NO REGULARIZE
            # compute gradient
            gradient = (1 / N) * (xtrain.T @ error) + (1 / N) * regularization
            # update weights
            weight = weight - learning_rate * gradient
            # compute RMSE after weight update
            y_pred_updated = xtrain @ weight
            rmse = self.rmse(y_pred_updated, ytrain)
            loss_per_epoch.append(rmse)

        return weight, loss_per_epoch

    def ridge_fit_SGD(self, xtrain: np.ndarray, ytrain: np.ndarray,
        c_lambda: float, epochs: int=100, learning_rate: float=0.001) ->Tuple[
        np.ndarray, List[float]]:
        """		
        Fit a ridge regression model using stochastic gradient descent.
        Although there are many valid initializations, to pass the local tests
        initialize the weights with zeros.
        
        Args:
            xtrain: (N,1+D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance with a bias term
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float, value of regularization constant
            epochs: int, number of epochs
            learning_rate: float
        Return:
            weight: (1+D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step
        Hints:
            - RMSE loss should be recorded AFTER the gradient update in each iteration.
            - Keep in mind that the number of epochs is the number of complete passes
            through the training dataset. SGD updates the weight for one datapoint at
            a time. For each epoch, you'll need to go through all of the points.
            - You should avoid applying regularization to the bias term in the gradient update
        
        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.
        """
        N, D_plus1 = xtrain.shape
        weight = np.zeros((D_plus1, 1))
        loss_per_step = []

        for epoch in range(epochs):
            for i in range(N):
                xi = xtrain[i:i+1, :] # (1, D+1)
                yi = ytrain[i:i+1, :] # (1, 1)
                y_pred_i = xi @ weight # (1, 1)
                error = y_pred_i - yi # (1, 1)
                # regularization term 
                regularization = (c_lambda / N) * weight
                regularization[0, 0] = 0  # NO REGULARIZE
                # compute gradient
                gradient = xi.T @ error + regularization
                # update weights
                weight = weight - learning_rate * gradient
                # compute RMSE over the entire training set
                y_pred_full = xtrain @ weight
                rmse = self.rmse(y_pred_full, ytrain)
                loss_per_step.append(rmse)

        return weight, loss_per_step

    def ridge_cross_validation(self, X: np.ndarray, y: np.ndarray, kfold:
        int=5, c_lambda: float=100) ->List[float]:
        """		
        For each of the k-folds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the RMSE for each fold
        
        Args:
            X : (N,1+D) numpy array, where N is the number of instances
                and D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            kfold: int, number of folds you should take while implementing cross validation.
            c_lambda: float, value of regularization constant
        Returns:
            loss_per_fold: list[float], RMSE loss for each kfold
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - For kfold=5:
                split X and y into 5 equal-size folds
                use 80 percent for training and 20 percent for test
        """
        N = X.shape[0]
        fold_size = N // kfold
        loss_per_fold = []

        for k in range(kfold):
            # define start and end indices for the validation fold
            start = k * fold_size
            end = start + fold_size if k != kfold - 1 else N  # last fold

            # validation set
            X_val = X[start:end]
            y_val = y[start:end]

            # training set
            X_train = np.concatenate((X[:start], X[end:]), axis=0)
            y_train = np.concatenate((y[:start], y[end:]), axis=0)

            # fit the model using ridge regression closed-form solution
            weight = self.ridge_fit_closed(X_train, y_train, c_lambda)

            # predict on validation set
            y_pred = self.predict(X_val, weight)

            # compute RMSE
            rmse = self.rmse(y_pred, y_val)
            loss_per_fold.append(rmse)

        return loss_per_fold

    def hyperparameter_search(self, X: np.ndarray, y: np.ndarray,
        lambda_list: List[float], kfold: int) ->Tuple[float, float, List[float]
        ]:
        """
        FUNCTION PROVIDED TO STUDENTS

        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N, 1+D) numpy array, where N is the number of instances and
                D is the dimensionality of each instance with a bias term
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants (lambdas) to search from
            kfold: int, Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the average RMSE error achieved using the best_lambda
            error_list: list[float] list of average RMSE loss for each lambda value given in lambda_list
        """
        best_error = None
        best_lambda = None
        error_list = []
        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            mean_err = np.mean(err)
            error_list.append(mean_err)
            if best_error is None or mean_err < best_error:
                best_error = mean_err
                best_lambda = lm
        return best_lambda, best_error, error_list
