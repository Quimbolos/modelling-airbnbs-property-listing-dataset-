# %% 
import torch
from tabular_data import load_airbnb
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split

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

AirbnbNightlyPriceImageDataset = AirbnbDataset()
print(AirbnbNightlyPriceImageDataset[10])
print(len(AirbnbNightlyPriceImageDataset))

DataLoader(AirbnbNightlyPriceImageDataset, batch_size=16, shuffle=True)

BATCH_SIZE = 64

train_dataset, validation_dataset, test_dataset = random_split(range(30), [0.7, 0.15, 0.15], generator=torch.Generator().manual_seed(42))

dataloaders = {
    "train": torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    ),
    "validation": torch.utils.data.DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available()
    ),
    "test": torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=torch.cuda.is_available()
    ),
}
# %%
