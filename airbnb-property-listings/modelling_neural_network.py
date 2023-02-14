# %% 
import torch
from tabular_data import load_airbnb
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

# Preprocessing the DataSet
from tabular_data import load_airbnb
import numpy as np

X, y = load_airbnb()
X.drop(532, axis=0, inplace=True)
y.drop(532, axis=0, inplace=True)
X['guests'] = X['guests'].str.replace('\'','').astype(np.float64)
X['bedrooms'] = X['bedrooms'].str.replace('\'','').astype(np.float64)
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

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

dataset = AirbnbNightlyPriceImageDataset()
print(dataset[10])
print(len(dataset))

batch_size = 16

# Split the data 
train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))

# Create DataLoaders
train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader=DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# len(train_dataset), len(validation_dataset), len(test_dataset)

from torch.utils.tensorboard import SummaryWriter
torch.manual_seed(10)

# yaml file content
''' 
optimiser: SGD
lr: 0.001
hidden_layer_width: 32
depth: 5
'''
# Define function get_nn_config()
import yaml
def get_nn_config():
    with open('nn_config.yaml', 'r') as stream:
    # Converts yaml document to python object
        dictionary=yaml.safe_load(stream)
    return dictionary

# Retrieve config dictionary
nn_config = get_nn_config()

# Neural Networks Model - Updated with more Layers
class NeuralNetwork(torch.nn.Module):

    def __init__(self):
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

model = NeuralNetwork()
loss_fn = torch.nn.MSELoss()

# Train function with Tensorboard
def train(model, dataloader, test_dl, epochs=20):
    """pass a train and validaton dataloader """
    # Set optimiser with lr from nn_config
    if nn_config['optimiser'] == "SGD":
        optimiser = torch.optim.SGD(model.parameters(), lr=nn_config['lr'])

    elif nn_config['optimiser'] == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), lr=nn_config['lr'])

    elif nn_config['optimiser'] == "Adagrad":
        optimiser = torch.optim.Adagrad(model.parameters(), lr=nn_config['lr'])

    writer = SummaryWriter()

    for epoch in range(epochs):
        batch_idx = 0
        current_loss = 0.0
        for batch in dataloader:
            features, labels = batch
            features = features.to(torch.float32) # Convert torch into the right format
            labels = labels.to(torch.float32) # Convert torch into the right format
            prediction = model(features)
            loss = loss_fn(prediction,labels)
            loss.backward() 
            optimiser.step() # Optimiser step
            optimiser.zero_grad()
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls
            # writer.add_scalar("Loss - Task 4",ls, epoch)
        
        # print (f"currentnt loss {current_loss} and batch index {batch_idx}")
        # print(f'Loss after mini-batch  ({epoch + 1} : {current_loss // batch_idx}')
        writer.add_scalar('loss',current_loss / batch_idx , epoch)
        print("Loss", current_loss / batch_idx)
        batch_idx = 0
        current_loss = 0.0
        for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
            inputs = inputs.to(torch.float32) # Convert  into the right format
            targets = targets.to(torch.float32) # Convert  into the right format
            yhat = model(inputs)
            loss = loss_fn(yhat,targets)
            ls = loss.item()
            batch_idx += 1
            current_loss = current_loss + ls
        
        writer.add_scalar('loss', current_loss)
# %%
# %%
