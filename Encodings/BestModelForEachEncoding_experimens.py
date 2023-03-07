# Include sslearn folder
import os
import sys
import warnings

sys.path.append('../../../sslearn')
import gc
import pickle as pkl
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SequenceEncoding import SequenceEncoding
from sklearn.base import TransformerMixin, clone
from sklearn.ensemble import (AdaBoostClassifier, RandomForestClassifier,
                              StackingClassifier)
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

def lstm(X):
    model = Sequential()
    model.add(LSTM(32, input_shape=(X.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



def main(enc, X, y, subsample_size, classifier, results_folder):
    
    # Change regression labels to binary labels above first quartile and below
    original_y = y.copy()
    y = np.where(y >= np.percentile(y, 75), 1, 0).ravel()

    results_file = os.path.join(results_folder, f'pred_dict_{enc}_{subsample_size}_{classifier}.pickle')

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    if os.path.exists(results_file):
        print(f'Already computed {results_file}. Skipping...')
    else:
        # Stratified kfold
        skf = StratifiedKFold(n_splits=5, shuffle=True)

        # Initialize dicts to store results
        pred_dict = []
        
        k = 0
        for train_index, test_index in skf.split(X, y):
            k += 1

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            original_y_train, original_y_test = original_y[train_index], original_y[test_index]
            
            # Get unlabeled subsets
            if subsample_size < 1:
                # Stratified subsampling
                sss = StratifiedShuffleSplit(n_splits=1, test_size=subsample_size)
                remove_indexes, subsample_indexes = next(sss.split(X_train, y_train))
                X_train_subsample = np.delete(X_train, remove_indexes, axis=0)
                y_train_subsample = np.delete(y_train, remove_indexes, axis=0)
            else:
                # Full training set
                X_train_subsample = X_train
                y_train_subsample = y_train

            start = time.time()

            if classifier == "LSTM":
                model = lstm(X_train_subsample)
            else:
                model = clone(classifier)
            model.fit(X_train_subsample, y_train_subsample)
            model_y_proba = model.predict_proba(X_test)[:, 1]
            pred_dict.append({"y_proba": model_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_subsample)})

            # Print formatted taken time in hours, minutes and seconds
            print(f"\tExperiment with {enc} and {model} using {subsample_size*100}% labeled instances (k={k}) took {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")
        
        with open(results_file, 'wb') as handle:
            pkl.dump(pred_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

        # Free memory (it seems threads are waiting taking memory innecesarily)
        del X_train
        del X_test
        del y_train
        del y_test
        del original_y_train
        del original_y_test
        del X_train_subsample
        del y_train_subsample
    del X
    del y
    gc.collect()


if __name__ == "__main__":
    
    dataset_folder = sys.argv[1]
    dataset = dataset_folder.split('data/')[-1].split('/')[0]
    
    results_folder = f"results/best_model_for_each_encoding_{dataset}/"

    subsample_sizes = [1, 0.5, 0.25, 0.15, 0.1, 0.05, 0.03, 0.01]
    models = [DecisionTreeClassifier(),
              RandomForestClassifier(),
              LogisticRegression(max_iter=10000),
              MLPClassifier(max_iter=10000),
              AdaBoostClassifier(),
              SVC(kernel='linear', probability=True),
              "LSTM"]
    
    y_file = os.path.join(dataset_folder, dataset+"_y.pkl")

    y = pkl.load(open(y_file, 'rb'))

    encoding_names = ["One_hot", "One_hot_6_bit", "Binary_5_bit",
                      "Hydrophobicity_matrix", "Meiler_parameters", "Acthely_factors",
                      "PAM250", "BLOSUM62",
                      "Miyazawa_energies", "Micheletti_potentials",
                      "AESNN3", "ANN4D", "ProtVec"]
    
    encodings_dict = dict()
    
    i=0
    for encoding_name in encoding_names:
        
        start = time.time()
        encoding_file = os.path.join(dataset_folder, f'X_{encoding_name}.pkl')
        i+=1

        # If file does not exist, encode
        if not os.path.exists(encoding_file):
            print(f"--> {encoding_name} [{i}/{len(encoding_names)}]", end="", flush=True)
            X_file = os.path.join(dataset_folder, dataset+"_X.pkl")
            X = pkl.load(open(X_file, 'rb'))        
            enc_X = np.array([SequenceEncoding(encoding_name).get_encoding(seq) for seq in X])
            enc_X = enc_X.reshape(enc_X.shape[0], -1)
            # Save the encoding to a pickle file
            with open(encoding_file, 'wb') as handle:
                pkl.dump(enc_X, handle, protocol=pkl.HIGHEST_PROTOCOL) 
            print(f"\tsize={round(enc_X.nbytes/(1024*1024), 2)} MB\t time={time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}", flush=True)
        else:
            enc_X = pkl.load(open(encoding_file, 'rb'))
        
        encodings_dict[encoding_name] = enc_X
    
    print(f"* Total dict size: {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024), 2)} MB | {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024*1024), 2)} GB", flush=True)
    
    arguments = []
    for subsample_size in subsample_sizes:
        for model in models:
            arguments.extend([(enc, encodings_dict[enc], y, subsample_size, model, results_folder) for enc in encoding_names])
    
    n_cores = int(sys.argv[2])
    with Pool(n_cores) as pool:
        pool.starmap(main, arguments, chunksize=1)
