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
        X_train, X_validation: numpy.ndarray
            A set of numpy arrays containing the features of the model

        y_train, y_validation: pandas.core.series.Series
            A set of pandas series containing the targets/labels 
        
    '''

    np.random.seed(10)

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=12)

    return X_train, X_validation, y_train, y_validation


def obtain_metrics(y_pred_train, y_train, y_pred_test, y_test):

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
    print("F1 score:", f1_score(y_test, y_pred_test, average="macro"))
    print("Precision:", precision_score(y_test, y_pred_test, average="macro"))
    print("Recall:", recall_score(y_test, y_pred_test, average="macro"))
    print("Accuracy:", accuracy_score(y_test, y_pred_test))

    test_metrics = {
        "F1 score" : f1_score(y_test, y_pred_test, average="macro"),
        "Precision":  precision_score(y_test, y_pred_test, average="macro"),
        "Recall" :  recall_score(y_test, y_pred_test, average="macro"),
        "Accuracy" :  accuracy_score(y_test, y_pred_test)
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


def run_methods():

    X, y = import_and_standardize_data()

    X_train, X_validation, y_train, y_validation = split_data(X, y)

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


if __name__ == "__main__":

    run_methods()


# %%

# %%
