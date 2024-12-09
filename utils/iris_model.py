"""
iris_model.py

Author: Lander Combarro Exposito
Created: 2024/04/12
Last Modified: 2024/04/12
"""

import os
import pickle

import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def download_data(data_path):
    # Check if file already exists
    if os.path.exists(data_path):
        print(f"The data already exists in {data_path}. No action taken")
        return None

    print('Downloading data from scikit-learn...')
    iris = datasets.load_iris()
    df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df_iris['species'] = iris.target

    # Create directory if neccesary
    if not os.path.exists(os.path.dirname(data_path)):
        os.makedirs(os.path.dirname(data_path))
    
    # Save data locally
    df_iris.to_csv(data_path, index=False)
    print(f'Data downloaded and saved in {data_path}')
    
    return None


def load_or_initialize_model(data_path, model_path):
    """
    Load an existing model from disk or train a new model if none exists.
    """
    # Check if the model exists, if so, load it
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print('Model not found')
        X_train, X_val, y_train, y_val = load_data(data_path=data_path)
        model = train_model(X_train=X_train, y_train=y_train)      
        save_model(model, model_path)
    
    return model


def load_model(model_path):
    """
    Load existing model.
    """
    if os.path.exists(model_path):
        print(f'Loading existing model from {model_path}...')
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        print('Model loaded')

    return model


def load_data(data_path):
    """
    Load local data.
    """
    # Check if the dataset exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found in {data_path}')

    print(f'Loading {data_path} dataset...')
    df = pd.read_csv(data_path)
    print(f'Dataset loaded')
    X = df.iloc[:, :-1]
    y = df['species']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val

    
def train_model(X_train, y_train):
    """
    Train a new Logistic Regression model on the dataset.
    """
    print('Training new model...')    
    # Initialize and train the model
    # model = LogisticRegression(max_iter=200)
    model = RandomForestClassifier(min_samples_leaf=30)
    model.fit(X_train, y_train)
    print('Model trained')
    
    return model


def save_model(model, model_path):
    """
    Save the trained model to the specified path.
    """
    print('Saving the trained model...')
    
    # Create the directory if it doesn't exist
    directory = os.path.dirname(model_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(model_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f'Model trained and saved in {model_path}')
    
    return None
