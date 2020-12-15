import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('datasets/diabetes.csv').to_numpy()

def load(filepath):
    '''
    Loads data from a csv file as a normalized numpy arr
    normalized with sklearn.preprocessing.MinMaxScaler
    '''
    data = pd.read_csv(filepath).to_numpy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data


def load_with_strings(filepath, names=None, padsize=0, skippad=2):
    original = pd.read_csv(filepath, names=names)
    numerical = pd.get_dummies(original)
    if padsize != 0:
        while len(numerical.columns) < padsize:
            numerical.insert(len(numerical.columns)-skippad, 'inserted', 0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(numerical.to_numpy())

    return scaled_data