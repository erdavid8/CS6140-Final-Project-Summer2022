import pandas as pd
import numpy as np
import zipfile 

def upzip():
    ''' 
    Unzips test and train data from Airline_Passanger_Satification zip file
    '''
    zip = zipfile.ZipFile('../../Airline_Passenger_Satisfaction_dataset.zip')
    zip.extractall()


def load_data(train_data_dir, test_data_dir):
    ''' data: input features
        labels: output features
    '''
    df1 = pd.read_csv(train_data_dir)
    df2 = pd.read_csv(test_data_dir)
    df = pd.concat([df1, df2])
    #id is not a feature, just a way to track user input
    df.drop('id', axis=1, inplace=True)
    df.drop('Unnamed: 0', axis=1, inplace=True)
    return df 

load_data("train.csv", "test.csv")


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
