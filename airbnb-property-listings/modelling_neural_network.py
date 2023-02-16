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
import itertools
torch.manual_seed(10)

# Preprocessing the DataSet
X, y = load_airbnb()
X.drop(532, axis=0, inplace=True)
y.drop(532, axis=0, inplace=True)
X['guests'] = X['guests'].str.replace('\'','').astype(np.float64)
X['bedrooms'] = X['bedrooms'].str.replace('\'','').astype(np.float64)

# Dataset Class
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


# Train function (Includes a Loss function, Optimiser and Tensorboard visualization)
def train(train_dataloader, validation_dataloader, nn_config, epochs=20):
    """
        Trains the Neural Network Model built using nn_config and using the training DataLoader. Then, the model is evaluated using the validation_dataloader
        
        Parameters
        ----------
        train_dataloader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the training dataset
        
        validation_dataloader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the validation dataset
        
        nn_config: dict 
            A dictionary containing the neural network configuration(Optimiser/LearningRate/Width/Depth)   

        Returns
        -------
        model: __main__.NeuralNetwork
            A model from pythorch
        
        training_duration: float
            The time taken to train the model
        
        interference_latency: float
            The average time taken to make a prediction  
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

    # Initialise TensorBoard writer
    writer = SummaryWriter()

    # Start the training_duration timer
    timer_start = time.time()

    # Train the model
    for epoch in range(epochs):
        batch_idx = 0
        current_loss = 0.0
        for batch in train_dataloader:
            features, labels = batch
            features = features.to(torch.float32) # Convert torch into the right format
            labels = labels.to(torch.float32) # Convert torch into the right format
            prediction = model(features)
            loss = loss_fn(prediction,labels)
            loss.backward() 
            optimiser.step() 
            optimiser.zero_grad() 
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls

        # Write the cumulative training loss for each batch
        writer.add_scalar('training_loss',current_loss / batch_idx , epoch)
        print("Loss", current_loss / batch_idx)

        # evaluate the model on the validation set for each batch
        batch_idx = 0
        current_loss = 0.0
        prediction_time_list = []
        for features, labels in validation_dataloader:
            features = features.to(torch.float32) # Convert  into the right format
            labels = labels.to(torch.float32) # Convert  into the right format
            timer_start_ = time.time() # Start timer for interference_latency
            yhat = model(features)
            timer_end_ = time.time() # End timer for interference_latency
            batch_prediction_time = (timer_end_-timer_start_)/len(features) # Calculate interference_latency for each batch
            prediction_time_list.append(batch_prediction_time) # Store interference_latency for each batch
            loss = loss_fn(yhat,labels)
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls

        writer.add_scalar('validation_loss', current_loss / batch_idx , epoch)

    # End the training_duration timer
    timer_end = time.time()

    # Calculate training_duration timer
    training_duration = timer_end - timer_start

    # Calculate interference_latency
    interference_latency =  sum(prediction_time_list) / len(prediction_time_list)

    return model, training_duration, interference_latency

# Define function get_nn_config()
def get_nn_config():
    '''
        Retrieves the content of a yaml file and returns a dictionary with it's contents

        Parameters
        ----------
        None 

        Returns
        -------
        dictionary: dict
            A dictionary containing the nn configuration from a .yaml file       
    '''
    with open('nn_config.yaml', 'r') as stream:
    # Converts yaml document to python object
        dictionary=yaml.safe_load(stream)

    return dictionary

def generate_nn_configs():
    '''
        Generates a list of dictionaries with different neural networks models

        Parameters
        ----------
        None 

        Returns
        -------
        config_dict_list: list
            A list of dictionaries containing the neural network parameters (Optimiser, lr, hidden_layer_width and depth)   
    '''

    # Parameters to change are: Optimiser, lr, hidden_layer_width and depth
    combinations_dict = {
        'Optimisers':['SGD', 'Adam', 'Adagrad'],
        'lr':[0.001, 0.0001],
        'hidden_layer_width':[32, 64, 128, 256],
        'depth':[3,5,10]
    }

    config_dict_list = []
    # For every possible combination of the combinations_dict create a custom dictionary that is later stored in config_dict_list
    for iteration in itertools.product(*combinations_dict.values()):
        config_dict = {
            'optimiser': iteration[0],
            'lr': iteration[1],
            'hidden_layer_width': iteration[2],
            'depth': iteration[3]
        }
        config_dict_list.append(config_dict)

    return config_dict_list

def save_model(best_model, best_hyperparameters, best_metrics):
    '''
        Creates a models folder, then within the models' folder creates a regression folder and finally creates a last neural networks folder where it stores the model, a dictionary of its hyperparameters and a dictionary of its metrics
        
        Parameters
        ----------
        folder_name: str
            A string used to name the folder to be created
        
        best_model: pytorch model
            A model from pytorch
        
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
                json.dump(str(best_metrics), fp)

    return 

def calculate_metrics(best_model, train_loader, validation_loader, test_loader):

    '''
        Calculates the RMSE and the R_squared for the train, validation and testing datasets

        Parameters
        ----------
        
        best_model: pytorch model
            A model from pytorch
        
        train_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the training dataset
        
        validation_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the validation dataset

        test_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the test dataset
        
        Returns
        -------
        train_RMSE_loss: np.float32
            The RMSE loss obtained using the training dataset

        validation_RMSE_loss: np.float32
            The RMSE loss obtained using the validation dataset

        test_RMSE_loss: np.float32
            The RMSE loss obtained using the testing dataset

        train_R_squared: np.float32
            The R_squared loss obtained using the training dataset

        validation_R_squared: np.float32
            The R_squared obtained using the validation dataset

        test_R_squared: np.float32
            The R_squared obtained using the testing dataset
    '''
    
    # Calculate RMSE and R_squared metrics

    y_hat = [] # Predictions
    y = [] # Targets

    # Obtain predictions using the training dataset features
    for features, labels in train_loader:
            features = features.to(torch.float32) # Convert  into the right format
            labels = labels.to(torch.float32) # Convert  into the right format
            prediction = best_model(features) 
            y.append(labels.detach().numpy())
            y_hat.append(prediction.detach().numpy())

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    y_hat = np.concatenate(y_hat)

    # If the predictions include nan values, assign poor metrics to discard the model later
    if np.isnan(y_hat).any():
        train_RMSE_loss = 1000000
        train_R_squared = 0
    else: # Else, calculate RMSE and R^2
        train_RMSE_loss = mean_squared_error(y, y_hat)
        train_R_squared = (r2_score(y, y_hat))
        pass

    y_hat = [] # Predictions
    y = [] # Targets

    # Obtain predictions using the validation dataset features
    for features, labels in validation_loader:
            features = features.to(torch.float32) # Convert  into the right format
            labels = labels.to(torch.float32) # Convert  into the right format
            prediction = best_model(features) 
            y.append(labels.detach().numpy())
            y_hat.append(prediction.detach().numpy()) 

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    y_hat = np.concatenate(y_hat)

    # If the predictions include nan values, assign poor metrics to discard the model later
    if np.isnan(y_hat).any():
        validation_RMSE_loss = 1000000
        validation_R_squared = 0
    else: # Else, calculate RMSE and R^2
        validation_RMSE_loss = mean_squared_error(y, y_hat)
        validation_R_squared = (r2_score(y, y_hat))
        pass

    y_hat = [] # Predictions
    y = [] # Targets

    # Obtain predictions using the validation dataset features
    for features, labels in test_loader:
            features = features.to(torch.float32) # Convert  into the right format
            labels = labels.to(torch.float32) # Convert  into the right format
            prediction = best_model(features) 
            y.append(labels.detach().numpy())
            y_hat.append(prediction.detach().numpy()) 

    y = np.concatenate(y)
    y_hat = np.concatenate(y_hat)
    y_hat = np.concatenate(y_hat)

    # If the predictions include nan values, assign poor metrics to discard the model later
    if np.isnan(y_hat).any():
        test_RMSE_loss = 1000000
        test_R_squared = 0
    else: # Else, calculate RMSE and R^2
        test_RMSE_loss = mean_squared_error(y, y_hat)
        test_R_squared = (r2_score(y, y_hat))
        pass

    return train_RMSE_loss,validation_RMSE_loss,test_RMSE_loss, train_R_squared, validation_R_squared, test_R_squared


def find_best_nn(config_dict_list, train_loader, validation_loader, test_loader):
    '''
        Trains various Neural Network Models using the train function, calculates the metrics using the calculate_metrics and train functions, and finally saves the best model using the save_model function

        Parameters
        ----------
        config_dict_list: list
            A list of dictionaries containing the neural network parameters (Optimiser, lr, hidden_layer_width and depth)   

        train_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the training dataset
        
        validation_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the validation dataset

        test_loader: torch.utils.data.dataloader.DataLoader
            A DataLoader containing the test dataset
        
        Returns
        -------
        None
    
    '''
    # For each configuration, redefine the nn_model and the training function
    for i, (nn_config) in enumerate(config_dict_list):

        best_metrics_ = None

        # Get the hyperparemeters
        best_hyperparameters = nn_config

        # Train the NN model using the model, the dataloaders and nn_config file
        best_model, training_duration, interference_latency = train(train_loader,validation_loader, nn_config)

        # Calculate the metrics
        train_RMSE_loss, validation_RMSE_loss, test_RMSE_loss, train_R_squared, validation_R_squared, test_R_squared = calculate_metrics(best_model, train_loader, validation_loader, test_loader)

        best_metrics = {

        'RMSE_loss' : [train_RMSE_loss,validation_RMSE_loss,test_RMSE_loss],
        'R_squared' : [train_R_squared, validation_R_squared, test_R_squared],
        'training_duration' : training_duration,
        'inference_latency' : interference_latency,
    }
        # Store the metrics, config, and model:
        if best_metrics_ == None or best_metrics.get('R_squared')[1]>best_metrics_.get('R_squared')[1]:
            best_model_ = best_model
            best_hyperparameters_ = best_hyperparameters
            best_metrics_ = best_metrics

        if i >= 20:
            break

    save_model(best_model_, best_hyperparameters_, best_metrics_)

    print(best_metrics_, best_hyperparameters_)

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

    # Call the generte nn configs function to get the list of config dictionaries
    config_dict_list = generate_nn_configs()

    # Call the find best nn model function
    find_best_nn(config_dict_list, train_loader, validation_loader, test_loader)

# %%

