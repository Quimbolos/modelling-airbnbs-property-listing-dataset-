#%%
from tabular_data import load_airbnb
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

def mse_loss(y_hat, labels): # define the criterion (loss function)
    errors = y_hat - labels ## calculate the errors
    squared_errors = errors ** 2 ## square the errors
    mean_squared_error = sum(squared_errors) / len(squared_errors) ## calculate the mean 
    return mean_squared_error

def calculate_loss(model, X, y):
    return mse_loss(
        model.predict(X),
        y
    )

X, y = load_airbnb()

X.drop(532, axis=0, inplace=True)
y.drop(532, axis=0, inplace=True)

np.random.seed(10)

X['guests'] = X['guests'].str.replace('\'','').astype(np.float64)
X['bedrooms'] = X['bedrooms'].str.replace('\'','').astype(np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

linear_regression_model_SDGRegr = SGDRegressor()

linear_regression_model = linear_model.LinearRegression()

model = linear_regression_model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Number of samples in:")
print(f" - Training: {len(y_train)}")
print(f" - Validation: {len(y_validation)}")
print(f" - Testing: {len(y_test)}")

print(" ")

print("Loss after fit:")
print(f"Training loss after fit: {calculate_loss(model, X_train, y_train)}")
print(f"Validation loss after fit: {calculate_loss(model, X_validation, y_validation)}")
print(f"Test loss after fit: {calculate_loss(model, X_test, y_test)}")

print(" ")

print('final weights:', model.coef_)
print('final bias:', model.intercept_)

print(" ")

print("MSE: ")
print(metrics.mean_squared_error(y_test, y_pred))
# %%
y_pred[0:5]

# %%
y_test.head(5)

# %%
