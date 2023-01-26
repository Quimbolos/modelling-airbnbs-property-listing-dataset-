# %%
from tabular_data import load_airbnb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import joblib 
from joblib import dump, load
import json


def load_airbnb():
    '''
    Returns the features and labels from the clean tabular data in a tuple. It only includes numerical tabular data.

    Parameters
    ----------
    None

    Returns
    -------
    features, labels: tuple
        A tuple containing all the numerical features for the ML model and the target to predict (Category)
    '''
    
    current_directory = os.getcwd()
    csv_relative_directory = 'airbnb-property-listings/tabular_data/clean_tabular_data.csv'
    csv_directory = os.path.join(current_directory, csv_relative_directory)
    df = pd.read_csv(csv_directory)
    labels = df['Category']
    features = df.drop(['ID','Category','Title','Description','Amenities','Location','Price_Night','url','Unnamed: 0'], axis=1)
    
    return features, labels


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
    scaled_features = std.fit_transform(X.values)
    X = pd.DataFrame(scaled_features, index=X.index, columns=X.columns)

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

    X_train, X_test, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=12)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_validation, test_size=0.5)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def obtain_metrics(y_pred, y):

    print("Metrics")
    print("F1 score:", f1_score(y, y_pred, average="macro"))
    print("Precision:", precision_score(y, y_pred, average="macro"))
    print("Recall:", recall_score(y, y_pred, average="macro"))
    print("Accuracy:", accuracy_score(y, y_pred))

    train_metrics = {
        "F1 score" : f1_score(y, y_pred, average="macro"),
        "Precision":  precision_score(y, y_pred, average="macro"),
        "Recall" :  recall_score(y, y_pred, average="macro"),
        "Accuracy" :  accuracy_score(y, y_pred)
    }

    return train_metrics


def classification_matrix(labels, predictions, clf):

    cm = confusion_matrix(labels, predictions, labels=clf.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    display.plot()
    plt.show()

    return

def normalised_classification_matrix(labels, predictions, clf):

    cm = confusion_matrix(labels, predictions, normalize='true', labels=clf.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    display.plot()
    plt.show()

    return

def tune_classification_model_hyperparameters(model, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict):
    '''
        Returns the best model, its metrics and the best hyperparameters after hyperparameter tunning. The best model is chosen based on the computed validation RMSE.

        Parameters
        ----------
        model: sklearn.model
            An instance of the sklearn model
        
        X_train, X_validation, X_test: numpy.ndarray
            A set of numpy arrays containing the features of the model

        y_train, y_validation, y_test: pandas.core.series.Series
            A set of pandas series containing the targets/labels
        
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
    
    X = pd.concat([X_train, X_validation, X_test])
    y = pd.concat([y_train, y_validation, y_test])
    
    model = model
    hyperparameters = hyperparameters_dict
    grid_search = GridSearchCV(model, hyperparameters, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    best_hyperparameters_dict[model] = grid_search.best_params_
    best_metrics_dict[model] = grid_search.best_score_
    if best_regression_model is None or best_metrics_dict[model] > best_metrics_dict[best_regression_model]:
        best_regression_model = model
        best_hyperparameters = best_hyperparameters_dict[model]

    
    model = best_regression_model.fit(X,y)
    best_regression_model = model
    y_pred_validation = model.predict(X_validation)

    best_metrics = {
        "F1 score" : f1_score(y_validation, y_pred_validation, average="macro"),
        "Precision":  precision_score(y_validation, y_pred_validation, average="macro"),
        "Recall" :  recall_score(y_validation, y_pred_validation, average="macro"),
        "Accuracy" :  accuracy_score(y_validation, y_pred_validation)
    }

    return best_regression_model, best_hyperparameters, best_metrics

def metrics_and_classification_matrices(labels, predictions, fit_model):

    # Predict metrics using training and validation datasets
    obtain_metrics(predictions, labels)

    # Obtain Confusion Matrices
    classification_matrix(labels, predictions, fit_model)

    # Obtain Normalised Confusion Matrices
    normalised_classification_matrix(labels, predictions, fit_model)
  
    return

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

    # Create classification folder
    classification_dir = 'airbnb-property-listings/models/classification'
    current_dir = os.path.dirname(os.getcwd())
    regression_path = os.path.join(current_dir, classification_dir)
    if os.path.exists(regression_path) == False:
        os.mkdir(regression_path)

    # Create logistic_regression folder
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
        Imports and Standardizes the data, splits the dataset and finds the best-tuned model from the provided sklearn models and a range of its hyperparameters.       
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

        best_regression_model, best_hyperparameters_dict, best_metrics_dict = tune_classification_model_hyperparameters(models[i], X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict[i])

        # Print Results
        print(best_regression_model, best_hyperparameters_dict, best_metrics_dict)

        # Save the models in their corresponding folders
        folder_name= str(models[i])[0:-2]
        save_model(folder_name, best_regression_model, best_hyperparameters_dict, best_metrics_dict)
        
        y_pred = best_regression_model.predict(X_test)
        metrics_and_classification_matrices(y_test,y_pred,best_regression_model)

    return

models = [LogisticRegression(), DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier()]

hyperparameters_dict = [{ # LogisticRegression

    'C': [1.0],
    'class_weight': ['balanced',None],
    'dual': [True, False],
    'fit_intercept': [True, False],
    'intercept_scaling': [1],
    'max_iter': [100],
    'multi_class': ['auto', 'ovr', 'multinomial'],
    'n_jobs': [None],
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'random_state': [None],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'tol': [0.0001],
    'verbose': [0],
    'warm_start': [True, False]

},
                        { # DecisionTreeClassifier
    'ccp_alpha': [0.0],
    'class_weight': [None],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [0.1, 1, 5, None],
    'max_features': ['auto', 'sqrt', 'log2', None],
    'max_leaf_nodes': [None],
    'min_impurity_decrease': [0.0],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'min_weight_fraction_leaf': [0.0],
    'random_state': [None],
    'splitter': ['best', 'random']

},
                        { # RandomForestClassifier
    'bootstrap': [True, False],
    'ccp_alpha': [0.0],
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None],
    'max_features': [1.0,'sqrt', 'log2', None],
    'max_leaf_nodes': [None],
    'max_samples': [None],
    'min_impurity_decrease': [0.0],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'min_weight_fraction_leaf': [0.0],
    'n_estimators': [50, 70, 100, 200],
    'n_jobs': [None],
    'oob_score': [True, False],
    'random_state': [None],
    'verbose': [0],
    'warm_start': [True, False]
},
                        { # GradientBoostingClassifier

    'ccp_alpha': [0.0],
    'criterion': ['friedman_mse', 'squared_error'],
    'init': [None],
    'learning_rate': [0.1, 0.5, 1],
    'loss': ['log_loss', 'deviance', 'exponential'],
    'max_depth': [3],
    'max_features': ['auto', 'sqrt', 'log2',None],
    'max_leaf_nodes': [None],
    'min_impurity_decrease': [0.0],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'min_weight_fraction_leaf': [0.0],
    'n_estimators': [10, 50, 70, 100],
    'n_iter_no_change': [None],
    'random_state': [None],
    'subsample': [1.0],
    'tol': [0.0001],
    'validation_fraction': [0.1],
    'verbose': [0],
    'warm_start': [True, False]
}
]



if __name__ == "__main__":

    evaluate_all_models(models,hyperparameters_dict)

    

# %%


# %%
