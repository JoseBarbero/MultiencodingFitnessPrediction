# Include sslearn folder
import os
# Set export OPENBLAS_NUM_THREADS=1
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import sys
import warnings

sys.path.append('../../../../sslearn')
sys.path.append('../../Encodings')
import gc
import pickle as pkl
import time
from itertools import combinations

from concurrent.futures import ProcessPoolExecutor as Pool
import random 
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from SequenceEncoding import SequenceEncoding
from Bio import SeqIO
from sklearn.base import TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, StackingRegressor, RandomForestRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, RidgeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sslearn.base import OneVsRestSSLClassifier
from sslearn.wrapper import CoTraining
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import StandardScaler

from models.TriTrainingRegressor import TriTrainingRegressor
from models.MultiViewCoRegression import MultiviewCoReg

import mkl
mkl.set_num_threads(1)
import argparse

class ColumnExtractor(TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self

def fold(arguments):

    model, enc1, enc1_X_train, enc1_X_test, y_train, y_test, labeled_percentage, ss_method, pred_dicts_ct, k, original_y_test = arguments

    start = time.time()

    if labeled_percentage < 1:
        # Get labeled_percentage random indexed from enc1_X_train
        labeled_indexes = np.random.choice(len(y_train), int(labeled_percentage * len(y_train)), replace=False)
        unlabeled_indexes = np.setdiff1d(np.arange(len(y_train)), labeled_indexes)
    elif labeled_percentage == 1:
        labeled_indexes = np.arange(len(y_train))
        unlabeled_indexes = []

    
    # Flatten
    enc1_X_train = enc1_X_train.reshape(enc1_X_train.shape[0], -1)
    enc1_X_test = enc1_X_test.reshape(enc1_X_test.shape[0], -1)

    # Set unlabeled_indexes to -1 in y
    y_train_ct = np.copy(y_train)
    y_train_ct[unlabeled_indexes] = np.nan

    ct = model
    try:
        ct.fit(enc1_X_train, y_train_ct)
    except IndexError as e:
        print(f"IndexError: {e}")
        print(f"enc1_X_train.shape: {enc1_X_train.shape}")
        print(f"y_train_ct.shape: {y_train_ct.shape}")
        print(f"labeled_indexes.shape: {labeled_indexes.shape}")
        print(f"unlabeled_indexes.shape: {unlabeled_indexes.shape}")
    
    ct_y_proba = ct.predict(enc1_X_test)
    
    pred_dict_ct.append({"y_proba": ct_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train)})

    # Print formatted taken time in hours, minutes and seconds
    print(f"\tExperiment with {enc1} using {labeled_percentage*100}% labeled instances (k={k}) took {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")


def main(arguments):

    enc1, enc1_X_train, enc1_X_test, y_train, y_test, labeled_percentage, model, results_folder = arguments
    
    # Change regression labels to binary labels above first quartile and below
    original_y_train = y_train.copy()
    original_y_test = y_test.copy()

    pred_dict_ct = dict()

    ct_results_file = os.path.join(results_folder, f'pred_dict_ct_{enc1}_{labeled_percentage}.pickle')

    # Skip if results file exists with size > 0
    if os.path.exists(ct_results_file) and os.path.getsize(ct_results_file) > 0:
        print(f"Results file {ct_results_file} already exists and is not empty. Skipping...")
        return
    
    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Stratified kfold
    n_splits = 5

    # Initialize dicts to store results
    pred_dict_ct = []
    
    # We can't use a classic skf split because the train and test are defined by the number of variants
    # So we just run it n_splits times
    arguments = []
    for k in range(n_splits):
        arguments.append([model, enc1, enc1_X_train, enc1_X_test, y_train, y_test, labeled_percentage, ss_method, pred_dicts_ct, k, original_y_test])

    # Create k threads that call fold function
    with Pool(k) as pool:
        pool.map(fold, arguments)
        pool.close()
        pool.join() 

    # Save dicts to pickle files
    if not os.path.exists(ct_results_file):
        with open(ct_results_file, 'wb') as handle:
            pkl.dump(pred_dict_ct, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # Free memory (it seems threads are waiting taking memory innecesarily)
    del enc1_X_train
    del enc1_X_test
    del y_train
    del y_test
    gc.collect()


if __name__ == "__main__":

    # How to run:
    # python multiview_extrapolation_experiments.py data/iris/ --cpus 32 --trainvariants 1 2 --testvariants 3
    CLI=argparse.ArgumentParser()
    CLI.add_argument(
        "--data", 
        type=str
    )
    CLI.add_argument(
        "--cpus",
        type=int,
        default=32,
    )
    CLI.add_argument(
        "--model", 
        type=str
    )
    CLI.add_argument(
        "--trainvariants",  
        nargs="*",  
        type=int,
        default=[1],
        )
    CLI.add_argument(
        "--testvariants",
        nargs="*",
        type=int,
        default=[2],
    )
    
    dataset_folder = CLI.parse_args().data
    dataset = dataset_folder.split('data/')[-1].split('/')[0]

    # Set the train and test indexes
    train_variants = CLI.parse_args().trainvariants
    test_variants = CLI.parse_args().testvariants
    
    # Create model
    if CLI.parse_args().model == "TriTrainingRegressor":
        model = TriTrainingRegressor() # TODO Set parameters
    elif CLI.parse_args().model == "MultiviewCoReg":
        model = MultiviewCoReg() # TODO Set parameters
    else:
        raise ValueError("Model not supported")
    
    model_name = model.__class__.__name__

    # Create results folder
    experiments_id = f"multiview_extrapolation_experiments_trainedwith_{'_'.join(str(x) for x in train_variants)}_testedwith_{'_'.join(str(x) for x in test_variants)}_{dataset}_{model_name}"
    
    #labeled_percentages = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]
    labeled_percentages = [0.1, 0.01, 0.001]
    
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

    # Split X and y in a dict where the key is the number of variants and the value are que indexes of the samples in X with that number of variants
    # Count the number of variants between the wild type sequence and each sample
    wt_sequence_file = os.path.join(dataset_folder, dataset+"_wt.fasta")
    wt_sequence = SeqIO.read(wt_sequence_file, "fasta").seq
    variants_dict = dict()
    X_file = os.path.join(dataset_folder, dataset+"_X.pkl")
    X = pkl.load(open(X_file, 'rb'))
    for i, seq in enumerate(X):
        variants = sum([1 for i in range(len(seq)) if seq[i] != wt_sequence[i]])
        if variants in variants_dict:
            variants_dict[variants].append(i)
        else:
            variants_dict[variants] = [i]
    print(f"Summary of {dataset}:")
    print(f"\tNumber of samples: {len(X)}")
    # Sort variants_dict by key and print the number of samples with each number of variants
    for k, v in sorted(variants_dict.items()):
        print(f"\tNumber of samples with {k} variants: {len(v)}")
      
    if train_variants != test_variants:
        train_indexes = []
        for variant in train_variants:
            train_indexes.extend(variants_dict[variant])
        test_indexes = []
        for variant in test_variants:
            test_indexes.extend(variants_dict[variant])
    else:
        # Train/test by indexes from X size
        X_indexes = []
        for variant in train_variants:
            X_indexes.extend(variants_dict[variant])

        train_indexes = random.sample(X_indexes, int(len(X_indexes)*0.8))
        test_indexes = list(set(X_indexes) - set(train_indexes))
    # Only use a 10% of the test indexes (knn is very slow predicting)
    test_indexes = np.random.choice(test_indexes, int(len(test_indexes)*0.05), replace=False)

    results_folder = os.path.join("results", experiments_id)
    
    print(f"* Total dict size: {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024), 2)} MB | {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024*1024), 2)} GB", flush=True)
    arguments = []
    for labeled_percentage in labeled_percentages:
        arguments.extend([( enc1, 
                            encodings_dict[enc1][train_indexes],  # enc1_X_train
                            encodings_dict[enc1][test_indexes],   # enc1_X_test
                            y[train_indexes],                     # y_train
                            y[test_indexes],                      # y_test
                            labeled_percentage,
                            clone(model),
                            results_folder) for enc1 in encoding_names])
    print(f"* Total number of experiments: {len(arguments)}")
    print(f"* Number of cores: {CLI.parse_args().cpus}")
    print(f"* Training with variants of length {train_variants} with {len(train_indexes)} samples")
    print(f"* Testing with variants of length {test_variants} with {len(test_indexes)} samples")
    print(f"* Results will be saved in {results_folder}")
    print(f"* Starting experiments...")

    # To avoid unintented multithreading:
    # https://stackoverflow.com/questions/19257070/unintended-multithreading-in-python-scikit-learn/42124978#42124978
    # terminal: export OPENBLAS_NUM_THREADS=1
    # To know numpy/scipy config: https://stackoverflow.com/questions/9000164/how-to-check-blas-lapack-linkage-in-numpy-and-scipy
    n_cores = CLI.parse_args().cpus
    with Pool(n_cores) as pool:
        pool.map(main, arguments, chunksize=1)
