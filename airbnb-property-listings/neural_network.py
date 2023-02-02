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

# Linear Model
class LinearRegression(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Initialise the Parameters
        self.linear_layer = torch.nn.Linear(11,1) # 11 features, 1 label

    def forward(self, features):
        # Use the layers to process the features
        return self.linear_layer(features)

model = LinearRegression()

loss_fn = torch.nn.MSELoss() # This Loss function is better

# Train function with optimiser
def train(model, dataloader, epochs=100):

    optimiser = torch.optim.SGD(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        for batch in dataloader:
            features, labels = batch
            features = features.to(torch.float32) # Convert torch into the right format
            labels = labels.to(torch.float32) # Convert torch into the right format
            prediction = model(features)
            loss = loss_fn(prediction, labels)
            loss.backward() # What does this do? Populates the gradients?
            optimiser.step() # Optimiser step
            optimiser.zero_grad()
        print(loss.item())   
    return

train(model,train_loader)
# %%
# %%
