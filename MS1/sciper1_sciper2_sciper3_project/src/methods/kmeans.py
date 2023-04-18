import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters
        self.centers = 0

    def k_means(self, data, max_iter=100):
        """
        Main K-Means algorithm that performs clustering of the data.
        
        Arguments: 
            data (array): shape (N,D) where N is the number of data samples, D is number of features.
            max_iter (int): the maximum number of iterations
        Returns:
            centers (array): shape (K,D), the final cluster centers.
            cluster_assignments (array): shape (N,) final cluster assignment for each data point.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        # Initialize the centers
        self.centers = init_centers(data, K)
    
        # Loop over the iterations
        for i in range(max_iter):

            if ((i+1) % 10 == 0):
                print(f"Iteration {i+1}/{max_iter}…")

            old_centers = centers.copy()  # keep in memory the centers of the previous iteration

            ### WRITE YOUR CODE HERE
            new_centers = compute_centers(x_train, cluster_assignments, K)

            # End of the algorithm if the centers have not moved
            if old_centers == new_centers:  ### WRITE YOUR CODE HERE
                print(f"K-Means has converged after {i+1} iterations!")
                break
        
        self.centers = new_centers    
        # Compute the final cluster assignments
        distances = compute_distance(x_train, self.centers)
        cluster_assignments = find_closest_cluster(distances)

            # Initialize the centers
        centers = init_centers(data, K)
        
        # Compute the final cluster assignments
        return centers, cluster_assignments

    
    def fit(self, training_data, training_labels):
        """
        Train the model and return predicted labels for training data.

        You will need to first find the clusters by applying K-means to
        the data, then to attribute a label to each cluster based on the labels.
        
        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): labels of shape (N,)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##

        centers, clusters = self.k_means(training_data, self.K)
        self.centers = centers
        center_label = assign_labels_to_centers(centers, clusters, training_labels)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data given the cluster center and their labels.

        To do this, first assign data points to their closest cluster, then use the label
        of that cluster as prediction.
        
        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE! 
        ###
        ##
        return pred_labels
    def compute_centers(data, cluster_assignments, K):
        """
        Compute the center of each cluster based on the assigned points.

        Arguments: 
            data: data array of shape (N,D), where N is the number of samples, D is number of features
            cluster_assignments: the assigned cluster of each data sample as returned by find_closest_cluster(), shape is (N,)
            K: the number of clusters
        Returns:
            centers: the new centers of each cluster, shape is (K,D) where K is the number of clusters, D the number of features
        """
        ### WRITE YOUR CODE HERE
    
        return centers
    
    def init_centers(data, K):
        ### WRITE YOUR CODE HERE
        # Select the first K random index
        random_idx = np.random.permutation(data.shape[0])[:K]
        # Use these index to select centers from data
        centers = data[random_idx[:K]]
        
        return centers
    
    def assign_labels_to_centers(centers, cluster_assignments, true_labels):
        """
        Use voting to attribute a label to each cluster center.

        Arguments: 
            centers: array of shape (K, D), cluster centers
            cluster_assignments: array of shape (N,), cluster assignment for each data point.
            true_labels: array of shape (N,), true labels of data
        Returns: 
            cluster_center_label: array of shape (K,), the labels of the cluster centers
        """
        ### WRITE YOUR CODE HERE
        cluster_center_label = np.zeros(centers.shape[0])
        for i in range(len(centers)):
            label = np.argmax(np.bincount(true_labels[cluster_assignments == i]))
            cluster_center_label[i] = label
        return cluster_center_label