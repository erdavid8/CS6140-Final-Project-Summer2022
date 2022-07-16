import pandas as pd
import numpy as np

def load_data(data_dir):
    ''' data: input features
        labels: output features
    '''

    info = np.genfromtxt(data_dir, delimiter=',')[1:, :]
    data = info[:, :-1]
    labels = info[:, -1:]
    
    return data, labels

def normalize_data(data_dir):
    ''' data: input features
        labels: output features
    '''
    data, labels = load_data(data_dir)
    new_data = np.zeros(data.shape)
    for feature in range(data.shape[1]):
        feature_v = data[:, feature]
        feature_v = (feature_v - feature_v.min()) / (feature_v.max() - feature_v.min())
        new_data[:, feature] = feature_v
    return data, labels
