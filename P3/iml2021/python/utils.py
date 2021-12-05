#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Antonio Sutera & Yann Claes

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from tsfresh.feature_extraction import extract_features

def load_data(data_path):

    FEATURES = range(2, 33)
    N_TIME_SERIES = 3500

    # Create the training and testing samples
    LS_path = os.path.join(data_path, 'LS')
    TS_path = os.path.join(data_path, 'TS')
    X_train, X_test = [np.zeros((N_TIME_SERIES, (len(FEATURES) * 512))) for i in range(2)]

    for f in FEATURES:
        data = np.loadtxt(os.path.join(LS_path, 'LS_sensor_{}.txt'.format(f)))
        X_train[:, (f-2)*512:(f-2+1)*512] = data
        data = np.loadtxt(os.path.join(TS_path, 'TS_sensor_{}.txt'.format(f)))
        X_test[:, (f-2)*512:(f-2+1)*512] = data
    
    y_train = np.loadtxt(os.path.join(LS_path, 'activity_Id.txt'))
    subject_train = np.loadtxt(os.path.join(LS_path, 'subject_Id.txt'))

    print('X_train size: {}.'.format(X_train.shape))
    print('y_train size: {}.'.format(y_train.shape))
    print('X_test size: {}.'.format(X_test.shape))

    return X_train, y_train, X_test, subject_train


def write_submission(y, where, submission_name='toy_submission.csv'):

    os.makedirs(where, exist_ok=True)

    SUBMISSION_PATH = os.path.join(where, submission_name)
    if os.path.exists(SUBMISSION_PATH):
        os.remove(SUBMISSION_PATH)

    y = y.astype(int)
    outputs = np.unique(y)

    # Verify conditions on the predictions
    if np.max(outputs) > 14:
        raise ValueError('Class {} does not exist.'.format(np.max(outputs)))
    if np.min(outputs) < 1:
        raise ValueError('Class {} does not exist.'.format(np.min(outputs)))
    
    # Write submission file
    with open(SUBMISSION_PATH, 'a') as file:
        n_samples = len(y)
        if n_samples != 3500:
            raise ValueError('Check the number of predicted values.')

        file.write('Id,Prediction\n')

        for n, i in enumerate(y):
            file.write('{},{}\n'.format(n+1, int(i)))

    print('Submission {} saved in {}.'.format(submission_name, SUBMISSION_PATH))
    

def extract_and_save(X, parameters, folder="LS/extracted_features"):
    splitted_x = np.split(X, 31, axis=1)
    for i in range(len(splitted_x)):
        df = pd.DataFrame(splitted_x[i])
        df["id"] = df.index
        df = df.melt(id_vars="id", var_name="time").sort_values(["id", "time"]).reset_index(drop=True)
        X = extract_features(df, column_id="id", column_sort="time", default_fc_parameters=parameters)
        X.to_csv("data/{}/sensor_{}.csv".format(folder, i))

def get_features_from(folder="LS/extracted_features"):
    features = []
    for i in range(31):
        t = pd.read_csv("data/{}/sensor_{}.csv".format(folder, i))
        features.append(np.array(t).T[1:].T)

    X_out = np.concatenate(features, axis=1)
    return X_out


if __name__ == '__main__':

    # Directory containing the data folders
    DATA_PATH = 'data'
    X_train, y_train, X_test = load_data(DATA_PATH)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train, y_train)

    y_test = clf.predict(X_test)

    write_submission(y_test, 'submissions')