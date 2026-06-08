import numpy as np
from kmeans import KMeans
from numpy.linalg import LinAlgError
from tqdm import tqdm # type: ignore
SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = True


class GMM(object):

    def __init__(self, X, K, max_iters=100):
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        self.N = self.points.shape[0]
        self.D = self.points.shape[1]
        self.K = K

    def softmax(self, logit):
        """		
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        Hint:
            Add keepdims=True in your np.sum() function to avoid broadcast error.
        """
        expLogit = np.exp(logit - np.max(logit, axis=1, keepdims=True)) # compute exponentials
        sumExpLogit = np.sum(expLogit, axis=1, keepdims=True) # take the sum
        return expLogit / sumExpLogit

    def logsumexp(self, logit):
        """		
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        Hint:
            The keepdims parameter could be handy
        """
        maxlogit = np.max(logit, axis=1, keepdims=True)  # N x 1
        expLogit = np.exp(logit - maxlogit)  # N x D
        sumExpLogit = np.sum(expLogit, axis=1, keepdims=True)  # N x 1
        return maxlogit + np.log(sumExpLogit)  # N x 1

    def normalPDF(self, points, mu_i, sigma_i):
        """		
        Args:
            points: N x D numpy array
            mu_i: (D,) numpy array, the center for the ith gaussian.
            sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
        
        Hint:
            np.diagonal() should be handy.
        """
        
        # normalization const
        const = 1.0 / (np.power(2 * np.pi, points.shape[1] / 2) * np.sqrt(np.prod(np.diagonal(sigma_i))))
        
        # Mahalanobis dist ^ 2
        mahalanobis_squared = np.sum(((points - mu_i) ** 2) / np.diagonal(sigma_i), axis=1)  # Shape: (N,)
        
        # exponent term
        exp_term = np.exp(-0.5 * mahalanobis_squared)  # N x ,
        
        # compute pdf
        return const * exp_term  # N x ,

    def multinormalPDF(self, points, mu_i, sigma_i): # DO NOT DO
        raise NotImplementedError # GRADS ONLY

    def create_pi(self):
        """		
        Initialize the prior probabilities
        Args:
        Return:
        pi: numpy array of length K, prior
        """
        return np.ones(self.K) / self.K

    def create_mu(self):
        """		
        Intialize random centers for each gaussian
        Args:
        Return:
        mu: KxD numpy array, the center for each gaussian.
        """
        N, D = self.points.shape
        indexes = np.random.choice(N, self.K, replace=True)
        return self.points[indexes]

    def create_sigma(self):
        """		
        Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
        by K diagonal matrices.
        Args:
        Return:
        sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
            You will have KxDxD numpy array for full covariance matrix case
        """
        N, D = self.points.shape
        return np.array([np.eye(D) for _ in range(self.K)])

    def _init_components(self, **kwargs):
        """		
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        
            Hint: np.random.seed(5) must be used at the start of this function to ensure consistent outputs.
        """
        np.random.seed(5)
        return self.create_pi(), self.create_mu(), self.create_sigma()

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """		
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        
        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
        """
        N, D = self.points.shape
        ll = np.zeros((N, self.K))
        for k in range(self.K):
            log_pi_k = np.log(pi[k] + LOG_CONST)
            pdf = self.normalPDF(self.points, mu[k], sigma[k])
            log_pdf = np.log(pdf + LOG_CONST)
            ll[:, k] = log_pi_k + log_pdf
        return ll


    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """		
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        
        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        ll = self._ll_joint(pi, mu, sigma, full_matrix=full_matrix)
        return self.softmax(ll)

    def _M_step(self, tau, full_matrix=FULL_MATRIX, **kwargs):
        """		
        Args:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
        
        Hint:
            There are formulas in the slides and in the Jupyter Notebook.
            Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
        """
        N, D = self.points.shape
        N_k = np.sum(tau, axis=0) + LOG_CONST  # K x ,

        pi = N_k / N  # K x ,
        mu = (tau.T @ self.points) / N_k[:, np.newaxis]  # K x D
        sigma = np.zeros((self.K, D, D))
        
        for k in range(self.K):
            # weighted sum
            weightedSquares = tau[:, k][:, np.newaxis] * ((self.points - mu[k]) ** 2)  # N x D
            sigma_k_diag = (1 / N_k[k]) * np.sum(weightedSquares, axis=0) + SIGMA_CONST  # D x ,
            sigma[k] = np.diag(sigma_k_diag)
            
        return pi, mu, sigma

    def __call__(
        self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=1e-16, **kwargs
    ):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        prev_loss = None
        for it in pbar:
            # E-step
            tau = self._E_step(pi, mu, sigma, full_matrix)

            # M-step
            pi, mu, sigma = self._M_step(tau, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description("iter %d, loss: %.4f" % (it, loss))
        return tau, (pi, mu, sigma)


def cluster_pixels_gmm(image, K, max_iters=10, full_matrix=True):
    """	
    Clusters pixels in the input image
    
    Each pixel can be considered as a separate data point (of length 3),
    which you can then cluster using GMM. Then, process the outputs into
    the shape of the original image, where each pixel is its most likely value.
    
    Args:
        image: input image of shape(H, W, 3)
        K: number of components
        max_iters: maximum number of iterations in GMM. Default is 10
        full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
    Return:
        clustered_img: image of shape(H, W, 3) after pixel clustering
    
    Hints:
        What do mu and tau represent?
    """
    # mu - mean colors of clusters
    # tau - responsibilities
    
    H, W, _ = image.shape

    # reshape image
    pixels = image.reshape(-1, 3)

    # run GMM
    gmm = GMM(pixels, K, max_iters=max_iters)
    tau, (pi, mu, sigma) = gmm(full_matrix=full_matrix)

    # assign each pixel a cluster
    cluster_assignments = np.argmax(tau, axis=1)  # N x ,
    clustered_pixels = mu[cluster_assignments]

    # reshape and return
    return clustered_pixels.reshape(H, W, 3)