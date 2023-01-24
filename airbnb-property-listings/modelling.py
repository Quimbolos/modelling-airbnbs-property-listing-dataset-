#%%
from tabular_data import load_airbnb
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import itertools
import os
import joblib 
from joblib import dump, load
import json

def import_and_standardize_data():
    '''
        Imports the data through the load_airbnb() function and then standardises it

        Parameters
        ----------
        None

        Returns
        -------
        X: numpy.ndarray
            A numpy array containing the features of the model

        y: pandas.core.series.Series
            A pandas series containing the targets/labels 
            
        '''

    X, y = load_airbnb()

    X.drop(532, axis=0, inplace=True)
    y.drop(532, axis=0, inplace=True)

    X['guests'] = X['guests'].str.replace('\'','').astype(np.float64)
    X['bedrooms'] = X['bedrooms'].str.replace('\'','').astype(np.float64)

    std = StandardScaler()
    X = std.fit_transform(X)

    return X, y

def split_data(X, y):
    '''
        Splits the data into training, validating and testing data

        Parameters
        ----------
        X: numpy.ndarray
            A numpy array containing the features of the model

        y: pandas.core.series.Series
            A pandas series containing the targets/labels 

        Returns
        -------
        X_train, X_validation, X_test: numpy.ndarray
            A set of numpy arrays containing the features of the model

        y_train, y_validation, y_test: pandas.core.series.Series
            A set of pandas series containing the targets/labels 
        
    '''

    np.random.seed(10)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.5)

    print("Number of samples in:")
    print(f" - Training: {len(y_train)}")
    print(f" - Validation: {len(y_validation)}")
    print(f" - Testing: {len(y_test)}")

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def custom_tune_regression_model_hyperparameters(models, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict):
    '''
        Returns the best model, its metrics and the best hyperparameters after hyperparameter tunning. The best model is chosen based on the computed validation RMSE.

        Parameters
        ----------
        models: abc.ABCMeta
            A list of models from sklearn in their abc.ABCMeta format

        X_train, X_validation, X_test: numpy.ndarray
            A set of numpy arrays containing the features of the model

        y_train, y_validation, y_test: pandas.core.series.Series
            A set of pandas series containing the targets/labels
        
        hyperparameters_dict: list
            A list of dictionaries containing a range of hyperparameters for each model

        Returns
        -------
        best_regression_model: sklearn.model
            A model from sklearn
        
        best_hyperparameters_dict: dict
            A dictionary containing the optimal hyperparameters configuration
        
        best_metrics_dict: dict 
            A dictionary containing the test metrics obtained using the best model         
    '''
    # Models input format : models = [SGDRegressor, linear_model.LinearRegression]

    # Lists to store metrics, chosen Hypermarameters and the model for each iteration
    validation_RMSE = []
    validation_R2 = []
    model_hyperparameters_val = [] 
    model_val = []

    # For each model, select the model class and the hyperparameters dictionary
    for i in range(len(models)):
        model = models[i]
        hyperparameters_dict_ = hyperparameters_dict[i]

        # For each hyperparameter combination, create a model and store its metrics and hyperparameters
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


def tune_regression_model_hyperparameters(model, X, y, X_test, y_test, hyperparameters_dict):
    '''
        Returns the best model, its metrics and the best hyperparameters after hyperparameter tunning. The best model is chosen based on the computed validation RMSE.

        Parameters
        ----------
        model: sklearn.model
            An instance of the sklearn model
        
        X: numpy.ndarray
            A numpy array containing the features of the model

        y: pandas.core.series.Series
            A pandas series containing the targets/labels

        X_test: numpy.ndarray
            A numpy array containing the features of the model

        y_test: pandas.core.series.Series
            A pandas series containing the targets/labels
        
        hyperparameters_dict: dict
            A dictionary containing a range of hyperparameters 

        Returns
        -------
        best_regression_model: sklearn.model
            A model from sklearn
        
        best_hyperparameters_dict: dict
            A dictionary containing the optimal hyperparameters configuration
        
        best_metrics_dict: dict 
            A dictionary containing the test metrics obtained using the best model         
    '''
    best_regression_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}
    
    
    model = model
    hyperparameters = hyperparameters_dict
    grid_search = GridSearchCV(model, hyperparameters, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)
    best_hyperparameters_dict[model] = grid_search.best_params_
    best_metrics_dict[model] = grid_search.best_score_
    if best_regression_model is None or best_metrics_dict[model] > best_metrics_dict[best_regression_model]:
        best_regression_model = model
        best_hyperparameters = best_hyperparameters_dict[model]

    
    model = best_regression_model.fit(X,y)
    y_pred = model.predict(X_test)

    test_RMSE = (metrics.mean_squared_error(y_test, y_pred, squared=False))
    test_R2 = (metrics.r2_score(y_test, y_pred))
    best_metrics = {
    'RMSE' : test_RMSE,
    'R^2' : test_R2
    } 

    return best_regression_model, best_hyperparameters, best_metrics

def save_model(folder_name, best_model, best_hyperparameters, best_metrics):
    '''
        Creates a models folder, then within the models' folder creates a regression folder and finally creates a last folder-name folder where it stores the model, a dictionary of its hyperparameters and a dictionary of its metrics
        
        Parameters
        ----------
        folder_name: str
            A string used to name the folder to be created
        
        best_model: sklearn.model
            A model from sklearn
        
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

    # Create linear_regression folder
    folder_name_dir = os.path.join(regression_path,folder_name)
    current_dir = os.path.dirname(os.getcwd())
    folder_name_path = os.path.join(current_dir, folder_name_dir)
    if os.path.exists(folder_name_path) == False:
        os.mkdir(folder_name_path)

    # Save the model in a file called model.joblib
    joblib.dump(best_model, os.path.join(folder_name_path, 'model.joblib'))
   
    # Save the hyperparameters in a file called hyperparameters.json
    with open(os.path.join(folder_name_path, 'hyperparameters.json'), 'w') as fp:
            json.dump(best_hyperparameters, fp)

    # Save the metrics in a file called metrics.json
    with open(os.path.join(folder_name_path, 'metrics.json'), 'w') as fp:
            json.dump(best_metrics, fp)


    return


def evaluate_all_models(models,hyperparameters_dict):
    '''
        Imports and Standirizes the data, splits the dataset and finds the best-tuned model from the provided sklearn models and a range of its hyperparameters.       
        Finally, it saves the models, their metrics and their hyperparameters in their corresponding folders.
        
        Parameters 
        ----------
        models: list
            A list of models from sklearn 
        
        hyperparameters_dict: list
            A list of dictionaries containing a range of hyperparameters for each model

        Returns
        -------
        None
             
    '''

    # Import and standardize data
    X, y = import_and_standardize_data()

    # Split Data
    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)

    # Tune models hyperparameters using GirdSearchCV
    for i in range(len(models)):

        best_regression_model, best_hyperparameters_dict, best_metrics_dict = tune_regression_model_hyperparameters(models[i], X, y, X_test, y_test, hyperparameters_dict[i])

        # Print Results
        print(best_regression_model, best_hyperparameters_dict, best_metrics_dict)

        # Save the models in their corresponding folders
        folder_name= str(models[i])[0:-2]
        save_model(folder_name, best_regression_model, best_hyperparameters_dict, best_metrics_dict)

    return

def find_best_model(models):
    '''
        Using the metrics.json files produced in the evaluate_all_models(), this function iterates through the files to find the best metrics and output the best model, its hyperparameters and its metrics.

        Parameters 
        ----------
        models: list
            A list of models from sklearn 

        Returns
        -------
        best_regression_model: sklearn.model
            A model from sklearn
        
        best_hyperparameters_dict: dict
            A dictionary containing the optimal hyperparameters configuration
        
        best_metrics_dict: dict 
            A dictionary containing the test metrics obtained using the best model   
             
    '''

    # Find best metrics (best R^2 == highest score) within the libraries 
    best_regression_model = None
    best_hyperparameters_dict = {}
    best_metrics_dict = {}

    regression_dir = 'airbnb-property-listings/models/regression'
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, regression_dir)
    
    for i in range(len(models)):
        model_str = str(models[i])[0:-2]
        model_dir = os.path.join(regression_path, model_str)
        model = load(os.path.join(model_dir, 'model.joblib'))
        hyperparameters_path = open(os.path.join(model_dir, 'hyperparameters.json'))
        hyperparameters = json.load(hyperparameters_path)
        metrics_path = open(os.path.join(model_dir, 'metrics.json'))
        metrics = json.load(metrics_path)

        if best_regression_model is None or metrics.get("R^2") > best_metrics_dict.get("R^2"):
            best_regression_model = model
            best_hyperparameters_dict = hyperparameters
            best_metrics_dict = metrics

    return best_regression_model, best_hyperparameters_dict, best_metrics_dict


models = [SGDRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor()]


hyperparameters_dict = [{ # SGDRegressor Hyperparameters (Selection)

    'loss':['squared_error','huber', 'squared_epsilon_insensitive'],
    'penalty':['l2', 'l1', 'elasticnet', 'None'],
    'alpha':[0.0001, 0.001],
    'l1_ratio':[0.15, 0.2],
    'fit_intercept':[True, False],
    'max_iter' :[1000],
    'shuffle' :[True, False],
    'early_stopping':[True, False]

},
                        { # DecisionTreeRegressor Hyperparameters (Selection)
    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter':['best', 'random'],
    'max_features':['auto', 'sqrt', 'log2' , None]

},
                        { # RandomForestRegressor Hyperparameters (Selection)
    'n_estimators':[50, 100,],
    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'bootstrap':[True, False]
},

                        { # GradientBoostingRegressor Hyperparameters (Selection)
    'loss':['squared_error','huber', 'squared_epsilon_insensitive'],
    'learning_rate':[0.1, 0.2, 0.5],
    'n_estimators':[50, 100, 200,],
    'criterion':['squared_error', 'friedman_mse'],
    'max_features':['auto', 'sqrt', 'log2' , None],

}]

if __name__ == "__main__":

    evaluate_all_models(models, hyperparameters_dict)

    best_regression_model, best_hyperparameters_dict, best_metrics_dict = find_best_model(models)

    
    print("Best Regression Model:")
    print(best_regression_model)
    print("Hyperparameters:")
    print(best_hyperparameters_dict)
    print("Metrics:")
    print(best_metrics_dict)


# %%

