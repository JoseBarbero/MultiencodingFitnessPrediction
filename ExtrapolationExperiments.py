import numpy as np
import os
import pickle as pkl
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# New class to handle the datasets
class Dataset:
    def __init__(self, X, y, protein, variants_file):
        self.X = X
        self.y = y
        self.shape = X.shape
        self.protein = protein
        self.variants_file = variants_file

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM model
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def lstm(input_shape):
    model = Sequential()
    model.add(LSTM(32, input_shape=input_shape))
    model.add(Dropout(0.8))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    return model

data_dir = "../Data/"
output_data_dir = "data"
protein_variants_files = dict()
# List only folders in data folder
for folder in os.listdir(data_dir):
    if os.path.isdir(os.path.join(data_dir, folder)):
        variants = []
        # List every file in folder with fasta extension
        for file in os.listdir(os.path.join(data_dir, folder)):
            if file.endswith(".csv"):
                variants.append(file)
        protein_variants_files[folder] = variants

datasets = []
for protein, variants_files in protein_variants_files.items():
    for variants_file in variants_files:
        X = pkl.load(open(os.path.join(output_data_dir, protein, variants_file.split("_encoded")[0] + "_X.pkl"), "rb"))
        y = pkl.load(open(os.path.join(output_data_dir, protein, variants_file.split("_encoded")[0] + "_y.pkl"), "rb"))
        datasets.append(Dataset(X, y, protein, variants_file))


# X must be padded in order to have the same shape
# TODO maybe adding zeros at the end reduces the performance too much
max_seq_length = max([dataset.X.shape[1] for dataset in datasets])

# Pad every dataset to match max_seq_length in each instance
for dataset in datasets:
    if dataset.X.shape[1] != max_seq_length:
        pad = np.zeros((dataset.X.shape[0], max_seq_length - dataset.X.shape[1], 20))
        dataset.X = np.concatenate((dataset.X, pad), axis=1)

results_oneout = {}

# Concatenate every dataset but one that is going to be the test set
for test_set in datasets:
    X_train = np.concatenate([dataset.X for dataset in datasets if dataset != test_set], axis=0)
    y_train = np.concatenate([dataset.y for dataset in datasets if dataset != test_set], axis=0)
    X_test = test_set.X
    y_test = test_set.y
    print("Dataset", test_set.protein, test_set.variants_file)
    print("\tTrain", X_train.shape, y_train.shape)
    print("\tTest", X_test.shape, y_test.shape)
    start = time.time()
    # Reshape X to linear model
    #X_train = X_train.reshape(X_train.shape[0], -1)
    #X_test = X_test.reshape(X_test.shape[0], -1)

    # Train model
    #model = LinearRegression()
    model = lstm(X_train.shape[1:])
    model.fit(X_train, y_train)
    
    # Get MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    end = time.time()
    # Print time spent formatted in hours, minutes and seconds
    print("\tTime spent: %d:%02d:%02d" % (int(end - start) // 3600, int(end - start) // 60 % 60, int(end - start) % 60))
    print("\tMSE", mse)
    # Save results
    results_oneout[test_set.protein + "_" + test_set.variants_file.split("_encoded")[0]] = mse


results_single = {}

for dataset in datasets:
    
    X = dataset.X
    y = dataset.y

    # Train test split with sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print("Dataset", dataset.protein, dataset.variants_file)
    print("Train", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)

    # Reshape X to linear model
    #X_train = X_train.reshape(X_train.shape[0], -1)
    #X_test = X_test.reshape(X_test.shape[0], -1)

    # Train model
    #model = LinearRegression()
    model = lstm(X_train.shape[1:])
    model.fit(X_train, y_train)
    
    # Get MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Save results
    results_single[dataset.protein + "_" + dataset.variants_file.split("_encoded")[0]] = mse

results_mix = {}
# Concatenate every dataset but one that is going to be the test set and leave a part of this one to test
for test_set in datasets:
    X_train = np.concatenate([dataset.X for dataset in datasets if dataset != test_set], axis=0)
    y_train = np.concatenate([dataset.y for dataset in datasets if dataset != test_set], axis=0)
    
    # Train test split with sklearn
    X_test, X_train_toadd, y_test, y_train_toadd = train_test_split(test_set.X, test_set.y, test_size=0.3, random_state=42)

    # Add the rest of the train set to the train set
    X_train = np.concatenate((X_train, X_train_toadd), axis=0)
    y_train = np.concatenate((y_train, y_train_toadd), axis=0)

    print("Dataset", test_set.protein, test_set.variants_file)
    print("\tTrain", X_train.shape, y_train.shape)
    print("\tTest", X_test.shape, y_test.shape)
    start = time.time()
    # Reshape X to linear model
    #X_train = X_train.reshape(X_train.shape[0], -1)
    #X_test = X_test.reshape(X_test.shape[0], -1)

    # Train model
    #model = LinearRegression()
    model = lstm(X_train.shape[1:])
    model.fit(X_train, y_train)
    
    # Get MSE
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    end = time.time()
    # Print time spent formatted in hours, minutes and seconds
    print("\tTime spent: %d:%02d:%02d" % (int(end - start) // 3600, int(end - start) // 60 % 60, int(end - start) % 60))
    print("\tMSE", mse)
    # Save results
    results_mix[test_set.protein + "_" + test_set.variants_file.split("_encoded")[0]] = mse




# Save results to files
with open("results_oneout.pkl", "wb") as f:
    pkl.dump(results_oneout, f)

with open("results_single.pkl", "wb") as f:
    pkl.dump(results_single, f)

with open("results_mix.pkl", "wb") as f:
    pkl.dump(results_mix, f)