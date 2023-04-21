import numpy as np


class KMeans(object):
    """
    K-Means clustering class.

    We also use it to make prediction by attributing labels to clusters.
    """

    def __init__(self, K=10, max_iters=100):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            K (int): number of clusters
            max_iters (int): maximum number of iterations
        """
        self.K = K
        self.max_iters = max_iters
        self.centers = 0 # will be initialized after importing data
        self.center_label = 0 # will be initialized after importing data

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
        centers = init_centers(data, self.K)
    
        # Loop over the iterations
        for i in range(3):
            

            if ((i+1) % 10 == 0):
                print(f"Iteration {i+1}/{max_iter}…")

            ### WRITE YOUR CODE HERE
            distances = compute_distance(data, centers)
            cluster_assignments = find_closest_cluster(distances)
            new_centers = compute_centers(data, cluster_assignments, self.K)

            # End of the algorithm if the centers have not moved
            if np.array_equal(centers,new_centers):  ### WRITE YOUR CODE HERE
                print(f"K-Means has converged after {i+1} iterations!")
                break
            centers = new_centers
        cluster_assignments = find_closest_cluster(distances)
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
        
        self.centers, clusters = self.k_means(training_data, self.K)
        self.center_label = assign_labels_to_centers(self.centers, clusters, training_labels)
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
        distances = compute_distance(test_data, self.centers)
        cluster_assignments = find_closest_cluster(distances)

        # Convert cluster index to label
        pred_labels = self.center_label[cluster_assignments]
        return pred_labels

    
    
    
    


#Required functions 

def init_centers(data, K):
        ### WRITE YOUR CODE HERE
        # Select the first K random index
        random_idx = np.random.permutation(data.shape[0])[:K]
        # Use these index to select centers from data
        centers = data[random_idx[:K]]
        
        return centers
   
def compute_distance(data, centers):
    """
    Compute the euclidean distance between each datapoint and each center.
    
    Arguments:    
        data: array of shape (N, D) where N is the number of data points, D is the number of features (:=pixels).
        centers: array of shape (K, D), centers of the K clusters.
    Returns:
        distances: array of shape (N, K) with the distances between the N points and the K clusters.
    """
    N = data.shape[0]
    K = centers.shape[0]
    # Here, we will loop over the cluster
    distances = np.zeros((N, K))
    for k in range(K):
        # Compute the euclidean distance for each data to each center
        center = centers[k]
        distances[:, k] = np.sqrt(((data - center) ** 2).sum(axis=1))
        
    return distances

def find_closest_cluster(distances):
    """
    Assign datapoints to the closest clusters.
    
    Arguments:
        distances: array of shape (N, K), the distance of each data point to each cluster center.
    Returns:
        cluster_assignments: array of shape (N,), cluster assignment of each datapoint, which are an integer between 0 and K-1.
    """
    cluster_assignments = np.argmin(distances, axis=1)
    return cluster_assignments

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
    # Here, we will loop over the cluster   
    D = data.shape[1]
    centers = np.zeros((K,D))
    cnts = np.zeros(K)
    for i in range(len(cluster_assignments)):
        centers[cluster_assignments[i]] += data[i]
        cnts[cluster_assignments[i]]+=1
    for i in range(K):
        centers[i] /= cnts[i]
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
        if true_labels[cluster_assignments == i].size >0:
            label = np.argmax(np.bincount(true_labels[cluster_assignments == i]))
            cluster_center_label[i] = label
        else: #If there is the center without assigned data
            cluster_center_label[i] = 0
    return cluster_center_label