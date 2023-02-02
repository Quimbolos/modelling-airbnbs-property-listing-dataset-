# %% 
import torch
from tabular_data import load_airbnb
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as F

# Preprocessing the DataSet
X, y = load_airbnb()
X.drop(532, axis=0, inplace=True)
y.drop(532, axis=0, inplace=True)
X['guests'] = X['guests'].str.replace('\'','').astype(np.float64)
X['bedrooms'] = X['bedrooms'].str.replace('\'','').astype(np.float64)
class AirbnbDataset(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        self.X, self.y = X , y
    # Not dependent on index
    def __getitem__(self, index):
        features = torch.tensor(self.X.iloc[index])
        label = self.y.iloc[index]
        return (features, label)

    def __len__(self):
        return len(self.X)
    

dataset = AirbnbDataset()

# print(dataset[10])
# print(len(dataset))

batch_size = 20

train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))

train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader=DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# train_example = next(iter(train_loader))
# features, labels = train_example
# features = features.to(torch.float64)
# labels = labels.to(torch.float64)
# print(features.dtype, labels.dtype)

# print(features)
class LinearRegression(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # Initialise the Parameters
        self.linear_layer = torch.nn.Linear(11,1)

    def forward(self, features):
        # Use the layers to process the features
        return self.linear_layer(features)

model = LinearRegression()
# print(model(features))

def train(model, dataloader, epochs=10):
    for epoch in range(epochs):
        for batch in dataloader:
            features, labels = batch
            prediction = model(features)
            loss = F.mse_loss(prediction, labels)
            loss.backward()
            print(loss)
    return

train(model, train_loader)
# %%

# %%
X.columns
# %%
