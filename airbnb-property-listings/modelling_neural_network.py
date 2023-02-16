# %% 
import torch
from tabular_data import load_airbnb
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os
import time
import json
from torch.utils.tensorboard import SummaryWriter
import yaml
torch.manual_seed(10)

# Preprocessing the DataSet
X, y = load_airbnb()
X.drop(532, axis=0, inplace=True)
y.drop(532, axis=0, inplace=True)
X['guests'] = X['guests'].str.replace('\'','').astype(np.float64)
X['bedrooms'] = X['bedrooms'].str.replace('\'','').astype(np.float64)

# DataSet Class
class AirbnbNightlyPriceImageDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = X , y
    # Not dependent on index
    def __getitem__(self, index):
        features = torch.tensor(self.X.iloc[index])
        label = torch.tensor(self.y.iloc[index])
        return (features, label)

    def __len__(self):
        return len(self.X)


# Neural Networks Model 
class NeuralNetwork(torch.nn.Module):

    def __init__(self, nn_config):
        super().__init__()
        # Initialise the Parameters
        #self.linear_layer = torch.nn.Linear(11,1) # 11 features, 1 label

        self.layers = torch.nn.Sequential()
        self.layers.add_module("Input Layer", torch.nn.Linear(11, nn_config['hidden_layer_width'])) # Input layer
        self.layers.add_module("ReLU", torch.nn.ReLU())
        for i in range(nn_config['depth'] - 2): #  The input and the first linear layer are already taken into account
            self.layers.add_module("Hidden Layer", torch.nn.Linear(nn_config['hidden_layer_width'], nn_config['hidden_layer_width'])) # Hidden Layer
            self.layers.add_module("Hidden ReLU", torch.nn.ReLU())
        self.layers.add_module("Output Layer", torch.nn.Linear(nn_config['hidden_layer_width'], 1))# output layer
    
    def forward(self, features):
        # Use the layers to process the features
        return self.layers(features)


# Train function with Tensorboard
def train( train_dataloader, test_dataloader, nn_config, epochs=20):
    """Creates a models folder, then within the models' folder creates a regression folder and finally creates a last neural networks folder where it stores the model, a dictionary of its hyperparameters and a dictionary of its metrics
        
        Parameters
        ----------
        model: str
            A string used to name the folder to be created
        
        train_dataloader: pytorch model
            A model from pythorch
        
        test_dataloader: dict
            A dictionary containing the optimal hyperparameters configuration
        
        nn_config: dict 
            A dictionary containing the test metrics obtained using the best model   

        Returns
        -------
        model: 
        
        training_duration:
        
        interference_latency:    
        
        """

    # Define the model
    model = NeuralNetwork(nn_config)

    # Define the loss function
    loss_fn = torch.nn.MSELoss() 

    # Set optimiser with lr from nn_config
    if nn_config['optimiser'] == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=nn_config['lr'])

    elif nn_config['optimiser'] == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=nn_config['lr'])

    elif nn_config['optimiser'] == "Adagrad":
        optimiser = torch.optim.Adagrad(model.parameters(), lr=nn_config['lr'])

    writer = SummaryWriter()

    # Start the timmer
    timer_start = time.time()
    for epoch in range(epochs):

        batch_idx = 0
        current_loss = 0.0
        for batch in train_dataloader:
            features, labels = batch
            features = features.to(torch.float32) # Convert torch into the right format
            labels = labels.to(torch.float32) # Convert torch into the right format
            # print (features.shape, labels.shape,"====== Train Shape" )
            # print (features[:1] , "the input feature array in Train")
            prediction = model(features)
            loss = loss_fn(prediction,labels)
            loss.backward() 
            optimiser.step() # Optimiser step
            optimiser.zero_grad()
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls
            # print ("The loss : ", ls)
        # print the cumulative loss for each batch
        writer.add_scalar('training_loss',current_loss / batch_idx , epoch)
        print("Loss", current_loss / batch_idx)
    
        batch_idx = 0
        current_loss = 0.0
        prediction_time_list = []

        for i, (features, labels) in enumerate(test_dataloader):
        # evaluate the model on the test set
            features = features.to(torch.float32) # Convert  into the right format
            labels = labels.to(torch.float32) # Convert  into the right format
            timer_start_ = time.time()
            yhat = model(features)
            timer_end_ = time.time()
            batch_prediction_time = (timer_end_-timer_start_)/len(features)
            prediction_time_list.append(batch_prediction_time)
            # time taken to predict for batch_size data points
            loss = loss_fn(yhat,labels)
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls

        writer.add_scalar('validation_loss', current_loss / batch_idx , epoch)

    # Calculate interference_latency
    interference_latency =  sum(prediction_time_list) / len(prediction_time_list)
    # for i, (time_) in enumerate(prediction_time_list):
    # interference_latency = 1 #(interference_latency + time_)/(i+1)

    timer_end = time.time()
    training_duration = timer_end - timer_start

    return model, training_duration, interference_latency

# Define function get_nn_config()
def get_nn_config():
    '''
    Creates a models folder, then within the models' folder creates a regression folder and finally creates a last neural networks folder where it stores the model, a dictionary of its hyperparameters and a dictionary of its metrics
        
        Parameters
        ----------
        None 

        Returns
        -------
        None       
    
    '''
    with open('nn_config.yaml', 'r') as stream:
    # Converts yaml document to python object
        dictionary=yaml.safe_load(stream)

    return dictionary

def save_model(best_model, best_hyperparameters, best_metrics):
    '''
        Creates a models folder, then within the models' folder creates a regression folder and finally creates a last neural networks folder where it stores the model, a dictionary of its hyperparameters and a dictionary of its metrics
        
        Parameters
        ----------
        folder_name: str
            A string used to name the folder to be created
        
        best_model: pytorch model
            A model from pythorch
        
        best_hyperparameters: dict
            A dictionary containing the optimal hyperparameters configuration
        
        best_metrics: dict 
            A dictionary containing the test metrics obtained using the best model   

        Returns
        -------
        None       
    '''

    # Create Models folder
    models_dir = 'airbnb-property-listings/models'
    current_dir = os.path.dirname(os.getcwd())
    models_path = os.path.join(current_dir, models_dir)
    if os.path.exists(models_path) == False:
        os.mkdir(models_path)

    # Create regression folder
    regression_dir = 'airbnb-property-listings/models/regression'
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)
    if os.path.exists(regression_path) == False:
        os.mkdir(regression_path)

    if str(type(best_model)) == "<class '__main__.NeuralNetwork'>":

        # Create neural_networks folder
        nn_name_dir = os.path.join(regression_path,'neural_networks') # Create the neural network folder
        current_dir = os.path.dirname(os.getcwd())
        nn_name_path = os.path.join(current_dir, nn_name_dir)
        if os.path.exists(nn_name_path) == False:
            os.mkdir(nn_name_path)

        # Create a Timestamp folder
        timestamp_dir = os.path.join(nn_name_dir,time.strftime("%Y-%m-%d_%H:%M:%S")) # Create the timestamp folder
        current_dir = os.path.dirname(os.getcwd())
        timestamp_path = os.path.join(current_dir, timestamp_dir)
        if os.path.exists(timestamp_path) == False:
            os.mkdir(timestamp_path)

        # Save the model in a file called model.pt
        torch.save(best_model, os.path.join(timestamp_path, 'model.pt')) 

        # Save the hyperparameters in a file called hyperparameters.json
        with open(os.path.join(timestamp_path, 'hyperparameters.json'), 'w') as fp: 
                json.dump(best_hyperparameters, fp)

        # Save the metrics in a file called metrics.json
        with open(os.path.join(timestamp_path, 'metrics.json'), 'w') as fp:
                json.dump(best_metrics, fp)

    return   


if __name__ == "__main__":

    # Define the DataSet
    dataset = AirbnbNightlyPriceImageDataset()

    # Define the batch size
    batch_size = 16

    # Split the data 
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))

    # Create DataLoaders
    train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader=DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)   

    # Retrieve config dictionary
    nn_config = get_nn_config()  


    # Train the NN model using the model, the dataloaders and nn_config file
    best_model, training_duration, interference_latency = train(train_loader,validation_loader, nn_config)

    # Get the hyperparemeters
    best_hyperparameters = get_nn_config()

    # Calculate the metrics
    y_hat = []
    y = []
    for i, (features, labels) in enumerate(train_loader):
        # evaluate the model on the test set
            features = features.to(torch.float32) # Convert  into the right format
            labels = labels.to(torch.float32) # Convert  into the right format
            prediction = best_model(features) 
            y.append(labels.detach().numpy())
            y_hat.append(prediction.detach().numpy())

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)

    train_RMSE_loss = mean_squared_error(y, y_hat)
    train_R_squared = (r2_score(y, y_hat))

    y_hat = []
    y = []
    for i, (features, labels) in enumerate(validation_loader):
        # evaluate the model on the test set
            features = features.to(torch.float32) # Convert  into the right format
            labels = labels.to(torch.float32) # Convert  into the right format
            prediction = best_model(features) 
            y.append(labels)
            y_hat.append(prediction.detach().numpy()) 

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)

    validation_RMSE_loss = (mean_squared_error(y, y_hat))
    validation_R_squared = (r2_score(y, y_hat))

    y_hat = []
    y = []
    for i, (features, labels) in enumerate(test_loader):
        # evaluate the model on the test set
            features = features.to(torch.float32) # Convert  into the right format
            labels = labels.to(torch.float32) # Convert  into the right format
            prediction = best_model(features) 
            y.append(labels)
            y_hat.append(prediction.detach().numpy()) 

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)

    test_RMSE_loss = (mean_squared_error(y, y_hat))
    test_R_squared = (r2_score(y, y_hat))

    best_metrics = {

        'RMSE_loss' : [train_RMSE_loss,validation_RMSE_loss,test_RMSE_loss],
        'R_squared' : [train_R_squared,validation_R_squared,test_R_squared],
        'training_duration' : training_duration,
        'inference_latency' : interference_latency,
    }

    # Save the model
    save_model(best_model, best_hyperparameters, best_metrics)
# %%
# %%
