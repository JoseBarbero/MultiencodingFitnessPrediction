# Include sslearn folder
import os
# Set export OPENBLAS_NUM_THREADS=1
os.environ['OPENBLAS_NUM_THREADS'] = '1'
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
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

import mkl
mkl.set_num_threads(1)
import argparse

def main(enc, enc_X_train, enc_X_test, y_train, y_test, labeled_percentage, model, results_folder, train_variants, test_variants):
    
    # Change regression labels to binary labels above first quartile and below
    original_y_train = y_train.copy()
    original_y_test = y_test.copy()
    if is_classifier(model):
        y_train = np.where(y_train >= np.percentile(y_train, 75), 1, 0).ravel()
        y_test = np.where(y_test >= np.percentile(y_test, 75), 1, 0).ravel()

    enc_results_file = os.path.join(results_folder, f'pred_dict_{enc}_{labeled_percentage}.pkl')

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    # Stratified kfold
    n_splits = 5
    if is_classifier(model):
        pred_mode = 'classification'
    elif is_regressor(model):
        pred_mode = 'regression'

    if hasattr(model, 'steps'):
        normalization_method = model.steps[1][0].__class__.__name__
    else:
        normalization_method = "None"

    # Initialize dicts to store results
    pred_dicts = []
    metadata_dict = {"Encoding": enc,
                     "Labeled percentage": labeled_percentage,
                     "Model": model.__class__.__name__,
                     "SS Method": "None",
                     "normalization_method": normalization_method,
                     "n_splits": n_splits,
                     "preds_file": enc_results_file,
                     "pred_mode": pred_mode,
                     "Train variants": train_variants,
                     "Test variants": test_variants}
    
    # We can't use a classic skf split because the train and test are defined by the number of variants
    # So we just run it n_splits times
    for k in range(n_splits):

        start = time.time()

        if is_classifier(model):
            # Get a stratifed sample with size = labeled_percentage
            # This is a bit strange. I am using StratifiedShuffleSplit to get labeled/unlabeled indexes
            # from enc_X_train and y_train. 
            # Then I am using those indexes to get the corresponding instances.
            # "test_size == labeled_percentage"
            if labeled_percentage < 1:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=labeled_percentage)
                unlabeled_indexes, labeled_indexes = next(sss.split(enc_X_train, y_train))
            # If labeled_percentage == 1, then we don't need to stratify
            elif labeled_percentage == 1:
                labeled_indexes = np.arange(len(enc_X_train))
                unlabeled_indexes = []
        elif is_regressor(model): # Regression can't be stratified bacause values are continuous
            if labeled_percentage < 1:
                # Get labeled_percentage random indexed from enc_X_train
                labeled_indexes = np.random.choice(len(enc_X_train), int(labeled_percentage * len(enc_X_train)), replace=False)
                unlabeled_indexes = np.setdiff1d(np.arange(len(enc_X_train)), labeled_indexes)
            elif labeled_percentage == 1:
                labeled_indexes = np.arange(len(enc_X_train))
                unlabeled_indexes = []

        # Get unlabeled subsets
        enc_X_train_onlylabeled = np.delete(enc_X_train, unlabeled_indexes, axis=0)
        y_train_onlylabeled = np.delete(y_train, unlabeled_indexes, axis=0)
        
        # Flatten
        enc_X_train = enc_X_train.reshape(enc_X_train.shape[0], -1)
        enc_X_train_onlylabeled = enc_X_train_onlylabeled.reshape(enc_X_train_onlylabeled.shape[0], -1)
        enc_X_test = enc_X_test.reshape(enc_X_test.shape[0], -1)
    

        # Model
        # If results file already exists, don't run model
        sup_model = clone(model)
        sup_model.fit(enc_X_train_onlylabeled, y_train_onlylabeled)
        if is_classifier(model):
            sup_model_y_proba = sup_model.predict_proba(enc_X_test)[:, 1]
        elif is_regressor(model):
            sup_model_y_proba = sup_model.predict(enc_X_test)
        else:
            raise ValueError('Model must be a classifier or regressor')
        pred_dicts.append({"y_proba": sup_model_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})
            
        # Print formatted taken time in hours, minutes and seconds
        print(f"\tExperiment with {enc} using {labeled_percentage*100}% labeled instances (k={k}) took {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")

    with open(enc_results_file, 'wb') as handle:
        pkl.dump(pred_dicts, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # Metadata
    with open(enc_results_file.replace(".pkl", "_metadata.pkl"), 'wb') as handle:
        pkl.dump(metadata_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


    # Free memory (it seems threads are waiting taking memory innecesarily)
    del enc_X_train
    del enc_X_test
    del enc_X_train_onlylabeled
    del y_train
    del y_train_onlylabeled
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
        "--normalize", 
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
    CLI.add_argument(
        "--enc",
        type=str,
        default="One_hot"
    )
    CLI.add_argument(
        "--labeled",
        nargs="*",
        type=float,
        default=[1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.0025, 0.001]
    )
    
    dataset_folder = CLI.parse_args().data
    dataset = dataset_folder.split('data/')[-1].split('/')[0]
    
    # Create model
    if CLI.parse_args().model == "DecisionTreeRegressor":
        model = DecisionTreeRegressor()
    elif CLI.parse_args().model == "Ridge":
        model = Ridge()
    elif CLI.parse_args().model == "RidgeClassifier":
        model = RidgeClassifier()
    elif CLI.parse_args().model == "LogisticRegression":
        model = LogisticRegression(max_iter=5000)
    elif CLI.parse_args().model == "KNeighborsRegressor":
        model = KNeighborsRegressor()
    elif CLI.parse_args().model == "SVR":
        model = SVR()
    else:
        raise ValueError("Model not supported")
    
    # Normalize
    if CLI.parse_args().normalize == "True":
        model = make_pipeline(StandardScaler(), model)
    
    if hasattr(model, 'steps'):
        model_name = '_'.join([submodel.__class__.__name__ for submodel in model])
    else:
        model_name = model.__class__.__name__

    # Create results folder
    experiments_id = f"supervised_extrapolation_experiments_{dataset}_{model_name}"
    
    labeled_percentages = CLI.parse_args().labeled
    
    y_file = os.path.join(dataset_folder, dataset+"_y.pkl")

    y = pkl.load(open(y_file, 'rb'))

    encoding_name = CLI.parse_args().enc

    start = time.time()
    encoding_file = os.path.join(dataset_folder, f'X_{encoding_name}.pkl')
    
    # If file does not exist, encode
    if not os.path.exists(encoding_file):
        print(f"--> Encoding {encoding_name}...", end="", flush=True)
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
      
    # Set the train and test indexes
    train_variants = CLI.parse_args().trainvariants
    test_variants = CLI.parse_args().testvariants

    if train_variants != test_variants:
        train_indexes = []
        for variant in train_variants:
            train_indexes.extend(variants_dict[variant])
        test_indexes = []
        for variant in test_variants:
            test_indexes.extend(variants_dict[variant])
    else:
        # Train/test by indexes from X size
        train_indexes = []
        for variant in train_variants:
            train_indexes.extend(variants_dict[variant])

        # Get a random sample of indexes for train (80%)
        train_indexes = np.random.choice(train_indexes, int(len(train_indexes)*0.8), replace=False)
        test_indexes = [i for i in range(len(X)) if i not in train_indexes]

    experiments_id = experiments_id.replace("experiments_", f"experiments_trainedwith_{'_'.join(str(x) for x in train_variants)}_testedwith_{'_'.join(str(x) for x in test_variants)}_")
    results_folder = os.path.join("results", experiments_id)

    arguments = []
    for labeled_percentage in labeled_percentages:
        arguments.append([encoding_name, 
                        enc_X[train_indexes],  # enc_X_train
                        enc_X[test_indexes],   # enc_X_test
                        y[train_indexes],      # y_train
                        y[test_indexes],       # y_test
                        labeled_percentage,
                        model,
                        results_folder,
                        train_variants,
                        test_variants])
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
        pool.starmap(main, arguments, chunksize=1)