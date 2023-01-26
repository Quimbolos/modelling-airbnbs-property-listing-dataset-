# %%
from tabular_data import load_airbnb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
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


def obtain_metrics(y_pred_train, y_train, y_pred_validation, y_validation):

    # Training
    print("Training")
    print("F1 score:", f1_score(y_train, y_pred_train, average="macro"))
    print("Precision:", precision_score(y_train, y_pred_train, average="macro"))
    print("Recall:", recall_score(y_train, y_pred_train, average="macro"))
    print("Accuracy:", accuracy_score(y_train, y_pred_train))

    train_metrics = {
        "F1 score" : f1_score(y_train, y_pred_train, average="macro"),
        "Precision":  precision_score(y_train, y_pred_train, average="macro"),
        "Recall" :  recall_score(y_train, y_pred_train, average="macro"),
        "Accuracy" :  accuracy_score(y_train, y_pred_train)
    }

    # Testing
    print("Testing")
    print("F1 score:", f1_score(y_validation, y_pred_validation, average="macro"))
    print("Precision:", precision_score(y_validation, y_pred_validation, average="macro"))
    print("Recall:", recall_score(y_validation, y_pred_validation, average="macro"))
    print("Accuracy:", accuracy_score(y_validation, y_pred_validation))

    test_metrics = {
        "F1 score" : f1_score(y_validation, y_pred_validation, average="macro"),
        "Precision":  precision_score(y_validation, y_pred_validation, average="macro"),
        "Recall" :  recall_score(y_validation, y_pred_validation, average="macro"),
        "Accuracy" :  accuracy_score(y_validation, y_pred_validation)
    }

    return test_metrics, train_metrics


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
    y_pred_validation = model.predict(X_validation)

    best_metrics = {
        "F1 score" : f1_score(y_validation, y_pred_validation, average="macro"),
        "Precision":  precision_score(y_validation, y_pred_validation, average="macro"),
        "Recall" :  recall_score(y_validation, y_pred_validation, average="macro"),
        "Accuracy" :  accuracy_score(y_validation, y_pred_validation)
    }

    return best_regression_model, best_hyperparameters, best_metrics

def run_methods():

    X, y = import_and_standardize_data()

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)

    clf = LogisticRegression(random_state=0).fit(X_train, y_train)

    # Predict labels using training and validation datasets
    y_pred_train = clf.predict(X_train)
    y_pred_validation = clf.predict(X_validation)

    # Predict metrics using training and validation datasets
    obtain_metrics(y_pred_train, y_train, y_pred_validation, y_validation)

    # Obtain Confusion Matrices
    classification_matrix(y_train, y_pred_train, clf)
    classification_matrix(y_validation, y_pred_validation, clf)

    # Obtain Normalised Confusion Matrices
    normalised_classification_matrix(y_train, y_pred_train, clf)
    normalised_classification_matrix(y_validation, y_pred_validation, clf)
    

    return

model = LogisticRegression()

hyperparameters_dict = {'C': [1.0],
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
 'warm_start': [True, False]}

if __name__ == "__main__":

    # run_methods()

    X, y = import_and_standardize_data()

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)

    best_regression_model, best_hyperparameters, best_metrics = tune_classification_model_hyperparameters(model, X_train, X_validation, X_test, y_train, y_validation, y_test, hyperparameters_dict)


# %%


# %%
