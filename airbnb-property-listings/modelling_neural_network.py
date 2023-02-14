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

# Neural Networks Model - Updated with more Layers
class NeuralNetwork(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Initialise the Parameters
        self.layers = torch.nn.Sequential( # Update Model with more Layers
        torch.nn.Linear(11, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
        )

    def forward(self, features):
        # Use the layers to process the features
        return self.layers(features)

model = NeuralNetwork()
loss_fn = torch.nn.MSELoss()

# Train function with Optimiser and Tensorboard
def train(model, dataloader, epochs=30):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.00001)

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
            print("Loss", ls)
            batch_idx += 1
            current_loss = current_loss + ls
            writer.add_scalar("Loss",ls, epoch)
        
        # print (f"currentnt loss {current_loss} and batch index {batch_idx}")
        # print(f'Loss after mini-batch  ({epoch + 1} : {current_loss // batch_idx}')
        # writer.add_scalar('loss',current_loss / batch_idx , epoch)
            
        
train(model,train_loader)

# %%
# %%
