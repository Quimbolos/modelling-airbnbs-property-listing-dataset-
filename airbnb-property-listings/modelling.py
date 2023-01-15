#%%
from tabular_data import load_airbnb
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import itertools

def custom_tune_regression_model_hyperparameters(models, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict):
    
    # Lists to store metrics, chosen Hypermarameters and the model for each iteration
    validation_RMSE = []
    validation_R2 = []
    model_hyperparameters_val = [] 
    model_val = []

    # For each model, select the model class and the hyperparameters dictionary
    for i in range(len(models)):
        model = models[i]
        hyperparameters_dict_ = hyperparameters_dict[i]

        # For each hyperparameter combination, create a model, and store it´s metrics and hyperparameters
        for hyperparameters in itertools.product(*hyperparameters_dict_.values()):
            hyperparameters_ = dict(zip(hyperparameters_dict_.keys(),hyperparameters))
            regression_model = model(**hyperparameters_)
            model_ = regression_model.fit(X_train, y_train)
            y_pred = model_.predict(X_validation)
            validation_RMSE.append(metrics.mean_squared_error(y_validation, y_pred, squared=False))
            validation_R2.append(metrics.r2_score(y_validation, y_pred))
            model_hyperparameters_val.append(hyperparameters_)
            model_val.append(regression_model)

    # Select the model with the best RMSE
    index = np.argmin(validation_RMSE)
    best_model = model_val[index]
    best_hyperparameters_dict = model_hyperparameters_val[index]

    # Train the best model
    best_regression_model = best_model.fit(X_train, y_train)
    y_pred_test= best_regression_model.predict(X_test)

    # Obtain the metrics
    test_RMSE = metrics.mean_squared_error(y_test, y_pred_test, squared=False)
    test_R2 = metrics.r2_score(y_test, y_pred_test)

    best_metrics_dict = {
        'RMSE' : test_RMSE,
        'R^2' : test_R2
    }

    return best_regression_model, best_hyperparameters_dict, best_metrics_dict

X, y = load_airbnb()

X.drop(532, axis=0, inplace=True)
y.drop(532, axis=0, inplace=True)

np.random.seed(10)

X['guests'] = X['guests'].str.replace('\'','').astype(np.float64)
X['bedrooms'] = X['bedrooms'].str.replace('\'','').astype(np.float64)

std = StandardScaler()
X = std.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

linear_regression_model_SDGRegr = SGDRegressor()
linear_regression_model = linear_model.LinearRegression()

model = linear_regression_model_SDGRegr.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)

print("Number of samples in:")
print(f" - Training: {len(y_train)}")
print(f" - Validation: {len(y_validation)}")
print(f" - Testing: {len(y_test)}")

print(" ")

print('final weights:', model.coef_)
print('final bias:', model.intercept_)

print(" ")
print("Testing Metrics")

print("MSE: ")
print(metrics.mean_squared_error(y_test, y_pred))

print("RMSE: ")
print(metrics.mean_squared_error(y_test, y_pred, squared=False))

print("MAE: ")
print(metrics.mean_absolute_error(y_test, y_pred))

print("R^2: ")
print(metrics.r2_score(y_test, y_pred))


print(" ")
print("Training Metrics")

print("MSE: ")
print(metrics.mean_squared_error(y_train, y_pred_train))

print("RMSE: ")
print(metrics.mean_squared_error(y_train, y_pred_train, squared=False))

print("MAE: ")
print(metrics.mean_absolute_error(y_train, y_pred_train))

print("R^2: ")
print(metrics.r2_score(y_train, y_pred_train))

# %%
models = [SGDRegressor, linear_model.LinearRegression]

hyperparameters_dict = [{

    'loss':['squared_error','huber', 'squared_epsilon_insensitive'],
    'penalty' : ['l2', 'l1', 'elasticnet', 'None'],
    'alpha' :[0.0001, 0.001]

},
                        {
    'fit_intercept':[True, False],
    'copy_X':[True, False],
    'n_jobs':[None],
    'positive':[True, False]

}]

custom_tune_regression_model_hyperparameters(models, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict)
    

# %%
