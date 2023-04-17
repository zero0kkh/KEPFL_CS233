import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, weights):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
            weights (array): weights of the logistic regression model, of shape(D, C)
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights = weights

    def f_softmax(data, weight):
        """
        Softmax function for multi-class logistic regression.
    
        Args:
            data (array): Input data of shape (N, D)
            weight (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
            range [0, 1] and each row sums to 1.
            The row i corresponds to the prediction of the ith data sample, and 
            the column j to the jth class. So element [i, j] is P(y_i=k | x_i, W)
        """
        z_arr = np.exp(data @ weight)
        z_sum = np.sum(z_arr, axis=1)
        sfx = z_arr
        for i in range(z_arr.shape[0]):
            sfx[i] = z_arr[i] / z_sum[i]
        return sfx
    
    """
    def loss_logistic_multi(self, data, labels, weight):
        #
        Loss function for multi class logistic regression, i.e., multi-class entropy.
    
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            weight (array): Weights of shape (D, C)
        Returns:
            float: Loss value 
        #
        preds = self.f_softmax(data, weight)
        return - np.sum(labels * np.log(preds) + (1-labels) * np.log(1 - preds))
    """

    def gradient(self, data, labels, weight):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
    
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            weight (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
    """
        return data.T @ (self.f_softmax(data, weight) - labels)
    
    def accuracy_fn(labels_pred, labels):
        """
        Computes the accuracy of the predictions (in percent).
    
        Args:
            labels_pred (array): Predicted labels of shape (N,)
            labels (array): Real labels of shape (N,)
        Returns:
            acc (float): Accuracy, in range [0, 100].
        """
        ### WRITE YOUR CODE HERE
        return np.sum(labels_pred == labels) / labels.shape[0] * 100.0

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        #### WRITE YOUR CODE HERE! 
        self.max_iters=10
        self.lr = 0.001
        
        D = training_data.shape[1]  # number of features
        C = 10  # number of classes

        # Random initialization of the weights
        self.weights = np.random.normal(0, 0.1, (D, C))
        
        for iter in range(self.max_iters):
            ### WRITE YOUR CODE HERE
            gradient = self.gradient(training_data, training_labels, self.weights)
            self.weights = self.weights - self.lr * gradient
            ##################################

            predictions = self.predict(training_data, self.weights) #(N,)
            if self.accuracy_fn(predictions, training_labels) == 100:
                break
        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ### WRITE YOUR CODE HERE
        predictions = self.f_softmax(test_data, self.weight) #(N,C)
        pred_labels = np.zeros(predictions.shape[0]) #(N,)
        for i in range(predictions.shape[0]):
            pred_labels[i] = np.argmax(predictions[i])
        return pred_labels #(N,)