import argparse
import numpy as np 
import torch
from torch.utils.data import DataLoader

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn
import time

def train_split(X, y, test_size=0.2, random_state=0):

    if not 0 < test_size < 1:
        raise ValueError("test_size must be a float between 0 and 1")
    if random_state:
        np.random.seed(random_state)
    data_size = len(X)
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    test_data_size = int(data_size * test_size)
    test_indices = indices[:test_data_size]
    train_indices = indices[test_data_size:]

    X_train = np.array([X[i] for i in train_indices])
    y_train = np.array([y[i] for i in train_indices])

    X_test = np.array([X[i] for i in test_indices])
    y_test = np.array([y[i] for i in test_indices])

    return X_train, X_test, y_train, y_test

def k_fold_split(X, y, k=5):
    n = len(X)
    fold_size = n // k
    indices = np.random.permutation(n)
    for i in range(k):
        test_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate((indices[:i * fold_size], indices[(i + 1) * fold_size:]))
        yield X[train_indices], y[train_indices], X[test_indices], y[test_indices]

def cross_validate(method_obj,xtrain,ytrain):
    accuracies = []
    macrof1s = []
    for X_train, y_train, X_test, y_test in k_fold_split(xtrain, ytrain):
        preds_train = method_obj.fit(X_train, y_train)
        preds = method_obj.predict(X_test)
        acc = accuracy_fn(preds, y_test)
        macrof1 = macrof1_fn(preds, y_test)
        accuracies.append(acc)
        macrof1s.append(macrof1)
    cross_acc=np.mean(accuracies)
    cross_f1=np.mean(macrof1s)
    return cross_acc, cross_f1

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)


    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.

    #  normalize
    xtrain = normalize_fn(xtrain,np.mean(xtrain),np.std(xtrain))
    xtest = normalize_fn(xtest,np.mean(xtest),np.std(xtest))

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        xtrain, xtest, ytrain, ytest = train_split(xtrain, ytrain, test_size=0.2, random_state=42)
        pass
    
    ### WRITE YOUR CODE HERE to do any other data processing


    # Dimensionality reduction (FOR MS2!)
    if args.use_pca:
        raise NotImplementedError("This will be useful for MS2.")
    

    ## 3. Initialize the method you want to use.
    if not args.method:
        args.method == "dummy_classifier"
        raise NotImplementedError("You have to give the method. This result is random")


    # Use NN (FOR MS2!)
    if args.method == "nn":
        raise NotImplementedError("This will be useful for MS2.")
    
    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":  ### WRITE YOUR CODE HERE
        method_obj = KMeans(K=args.K)
    elif args.method == "logistic_regression":
        method_obj = LogisticRegression(lr=args.lr, max_iters=args.max_iters)
    elif args.method == "svm":
        method_obj = SVM(C=args.svm_c, kernel=args.svm_kernel, degree=args.svm_degree, gamma=args.svm_gamma, coef0=args.svm_coef0)
    

    s1=time.time()
    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    preds_train = method_obj.fit(xtrain, ytrain)
        
    # Predict on unseen data
    preds = method_obj.predict(xtest)
    s2=time.time()
    print("This method takes ", s2-s1, "seconds")


    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    
    cross_acc, cross_f1 = cross_validate(method_obj,xtrain,ytrain)
    print(f"Cross-Validation of Test set:  accuracy = {cross_acc:.3f}% - F1-score = {cross_f1:.6f}")



if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    # Feel free to add more arguments here if you need!

    # Arguments for MS2
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")

    # "args" will keep in memory the arguments and their value,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
