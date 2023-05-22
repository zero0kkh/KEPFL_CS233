import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2
    
class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super().__init__() #for initialization of the nn.Module
        
        #### WRITE YOUR CODE HERE! 
        self.fc1 = nn.Linear(input_size, 512) #fc: fully connected layer
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, n_classes)
        ##########################
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        #### WRITE YOUR CODE HERE! 
        x = x.flatten(-3) #flatten the images into vectors
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = self.fc4(x)
        ##########################
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__() #for initialization of the nn.Module
        
        #### WRITE YOUR CODE HERE!
        filters = (8, 16, 32)
        
        #[1]    32*32 => 8*32*32, [2] 8*32*32 => 8*16*16, [3] 8*16*16 => 16*16*16
        #[4] 16*16*16 => 16*8*8 , [5]  16*8*8 => 32*8*8 , [6]  32*8*8 => 32*4*4
        #[7] 32*4*4 = 512 => 128, [8]   128   =>   64   , [9]    64   =>   20
        
        self.conv2d1 = nn.Conv2d(input_channels, filters[0], 3, 1, padding=1)  # [1]
        self.conv2d2 = nn.Conv2d(filters[0], filters[1], 3, 1, padding=1) # [3]
        self.conv2d3 = nn.Conv2d(filters[1], filters[2], 3, 1, padding=1) # [5]

        self.fc1 = nn.Linear(4 * 4 * filters[2], 128) # [7]
        self.fc2 = nn.Linear(128, 64)                 # [8]
        self.fc3 = nn.Linear(64, n_classes)           # [9]
        ##########################
        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        #### WRITE YOUR CODE HERE!
        x = F.max_pool2d(F.relu(self.conv2d1(x)), 2) # [2]
        x = F.max_pool2d(F.relu(self.conv2d2(x)), 2) # [4]
        x = F.max_pool2d(F.relu(self.conv2d3(x)), 2) # [6]
        x = x.flatten(-3)                            #flatten the images into vectors
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        preds = self.fc3(x)
        ##########################        
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr = lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)

            ### WRITE YOUR CODE HERE if you want to do add else at each epoch

    def train_one_epoch(self, dataloader):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        #### WRITE YOUR CODE HERE!
        self.model.train()
        for it, batch in enumerate(dataloader):
            # 5.1 Load a batch, break it down in images and targets.
            x, y = batch

            # 5.2 Run forward pass.
            logits = self.model(x)
            
            # 5.3 Compute loss (using 'criterion').
            loss = self.criterion(logits, y)
            
            # 5.4 Run backward pass.
            loss.backward()
            
            # 5.5 Update the weights using 'optimizer'.
            optimizer.step()
            
            # 5.6 Zero-out the accumulated gradients.
            optimizer.zero_grad()

            print('\rit {}/{}: loss train: {:.2f}, accuracy train: {:.2f}'.
                  format(it + 1, len(dataloader), loss,
                         accuracy(logits, y)), end='')
        ##########################  


    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        #### WRITE YOUR CODE HERE!
        self.model.eval()
        with torch.no_grad():
            acc_run = 0
            for it, batch in enumerate(dataloader):
                # Get batch of data.
                x, y = batch
                curr_bs = x.shape[0]
                preds = model(x)
                acc_run += accuracy(preds, y) * curr_bs
            acc = acc_run / len(dataloader.dataset)
            print('accuracy test: {:.2f}'.format(acc))
            
            pred_labels = onehot_to_label(preds)  # onehot_to_label() in utils.py
        ########################## 
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.numpy()