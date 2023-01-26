# %%
from tabular_data import load_airbnb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
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
        A tuple containing all the numerical features for the ML model and the target to predict (Price for the Night)
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


def run_methods():

    X, y = import_and_standardize_data()

    X_train, X_validation, X_test, y_train, y_validation, y_test = split_data(X, y)

    clf = LogisticRegression(random_state=0).fit(X, y)
    y_pred = clf.predict(X_validation)

    return 



if __name__ == "__main__":

    run_methods()


# %%

# %%
