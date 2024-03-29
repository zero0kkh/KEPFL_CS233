import matplotlib.pyplot as plt
import numpy as np

## MS2


class PCA(object):
    """
    PCA dimensionality reduction class.

    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d

        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

        # Additional part to check the explained variances of components
        self.eigvals = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT:
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##

        ### WRITE YOUR CODE BELOW ###
        # Compute the mean of data
        self.mean = np.mean(training_data, axis=0)
        # Center the data with the mean
        X_tilde = training_data - self.mean
        # Create the covariance matrix
        C = np.cov(X_tilde.T)
        # Compute the eigenvectors and eigenvalues. Hint: look into np.linalg.eigh()
        eigvals, eigvecs = np.linalg.eigh(C)
        # Hint: sort the eigenvalues (with corresponding eigenvectors) in decreasing order first.
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        self.eigvals = eigvals
        eg = np.sum(eigvals)

        # Choose the top d eigenvalues and corresponding eigenvectors
        eigvals = eigvals[: self.d]
        eigvecs = eigvecs[:, : self.d]

        self.W = eigvecs

        # Compute the explained variance
        exvar = np.sum(eigvals) / eg * 100
        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        data_centered = data - self.mean
        data_reduced = np.dot(data_centered, self.W)
        return data_reduced

    def plot_explained_variance(self):
        # Compute the explained variance for each component and cumulatively
        explained_variance = self.eigvals / np.sum(self.eigvals)
        cumulative_explained_variance = np.cumsum(explained_variance)

        plt.figure(figsize=(10, 5))

        # Plot the individual explained variance
        plt.bar(
            range(len(explained_variance)),
            explained_variance,
            alpha=0.5,
            align="center",
            label="individual explained variance",
        )

        # Plot the cumulative explained variance
        plt.step(
            range(len(cumulative_explained_variance)),
            cumulative_explained_variance,
            where="mid",
            label="cumulative explained variance",
        )
        plt.ylabel("Explained variance ratio")
        plt.xlabel("Principal components")
        plt.legend(loc="best")
        plt.grid()
        plt.show()
