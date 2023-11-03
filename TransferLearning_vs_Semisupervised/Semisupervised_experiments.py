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

#from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import Pool

from pathlib import Path
import random 
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
from sslearn.base import OneVsRestSSLClassifier
from sslearn.wrapper import CoTraining, SelfTraining, TriTraining
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import StandardScaler

from ss_code.models.MultiViewCoRegression import MultiviewCoReg
from ss_code.models.TriTrainingRegressor import TriTrainingRegressor

import mkl
mkl.set_num_threads(1)
import argparse

def fold(arguments):

    model, enc1, enc2, enc1_X_train, enc2_X_train, enc1_X_test, enc2_X_test, y_train, y_test, labeled_percentage, ss_method, pred_dicts_ct, k, original_y_test, temp_results_file = arguments
    
    np.random.seed(k)

    start = time.time()

    if is_classifier(model):
        # Get a stratifed sample with size = labeled_percentage
        # This is a bit strange. I am using StratifiedShuffleSplit to get labeled/unlabeled indexes
        # from enc1_X_train and y_train. 
        # Then I am using those indexes to get the corresponding instances.
        # "test_size == labeled_percentage"
        if labeled_percentage < 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=labeled_percentage)
            unlabeled_indexes, labeled_indexes = next(sss.split(enc1_X_train, y_train))
        # If labeled_percentage == 1, then we don't need to stratify
        elif labeled_percentage == 1:
            labeled_indexes = np.arange(len(enc1_X_train))
            unlabeled_indexes = []
    elif is_regressor(model): # Regression can't be stratified bacause values are continuous
        if labeled_percentage < 1:
            # Get labeled_percentage random indexed from enc1_X_train
            labeled_indexes = np.random.choice(len(enc1_X_train), int(labeled_percentage * len(enc1_X_train)), replace=False)
            unlabeled_indexes = np.setdiff1d(np.arange(len(enc1_X_train)), labeled_indexes)
        elif labeled_percentage == 1:
            labeled_indexes = np.arange(len(enc1_X_train))
            unlabeled_indexes = []

    # Get unlabeled subsets
    enc1_X_train_onlylabeled = np.delete(enc1_X_train, unlabeled_indexes, axis=0)
    enc1_X_train = enc1_X_train.reshape(enc1_X_train.shape[0], -1)
    enc1_X_train_onlylabeled = enc1_X_train_onlylabeled.reshape(enc1_X_train_onlylabeled.shape[0], -1)
    enc1_X_test = enc1_X_test.reshape(enc1_X_test.shape[0], -1)

    y_train_onlylabeled = np.delete(y_train, unlabeled_indexes, axis=0)

    if enc2:
        enc2_X_train_onlylabeled = np.delete(enc2_X_train, unlabeled_indexes, axis=0)
        enc2_X_train = enc2_X_train.reshape(enc2_X_train.shape[0], -1)
        enc2_X_train_onlylabeled = enc2_X_train_onlylabeled.reshape(enc2_X_train_onlylabeled.shape[0], -1)
        enc2_X_test = enc2_X_test.reshape(enc2_X_test.shape[0], -1)

    # SS training model
    if is_classifier(model):
        # Set unlabeled_indexes to -1 in y
        y_train_ct = np.copy(y_train)
        y_train_ct[unlabeled_indexes] = -1

        ct = ss_method(base_estimator=clone(model))   
        
        #ct.fit(rescaled_enc1_X_train, y_train_ct, X2=rescaled_enc2_X_train)

        if enc2 == None:
            ct.fit(enc1_X_train, y_train_ct)
            ct_y_proba = ct.predict_proba(enc_X_test)[:, 1]
        else:
            ct.fit(enc1_X_train, y_train_ct, X2=enc2_X_train)
            ct_y_proba = ct.predict_proba(enc1_X_test, X2=enc2_X_test)[:, 1]           
        
        pred_dicts_ct.append({"y_proba": ct_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})
    else:
        y_train_ct = np.copy(y_train)
        y_train_ct[unlabeled_indexes] = np.nan
        
        ct = ss_method

        if enc2 == None:
            ct.fit(enc1_X_train, y_train_ct)
            ct_y_pred = ct.predict(enc1_X_test)
        else:
            ct.fit(enc1_X_train, y_train_ct, X2=enc2_X_train)
            ct_y_pred = ct.predict(enc1_X_test, X2=enc2_X_test)

        pred_dicts_ct.append({"y_proba": ct_y_pred, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})
        
    # Save temporary results
    with open(temp_results_file, "wb") as f:
        pkl.dump(pred_dicts_ct, f)

    # Print formatted taken time in hours, minutes and seconds
    if enc2 == None:
        print(f"\tExperiment with {enc1} using {labeled_percentage*100}% labeled instances (k={k}) took {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")
    else:
        print(f"\tExperiment with {enc1} and {enc2} using {labeled_percentage*100}% labeled instances (k={k}) took {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")



def main(arguments):
    enc1, enc2, enc1_X_train, enc2_X_train, enc1_X_test, enc2_X_test, y_train, y_test, labeled_percentage, ss_method, model, results_folder, train_variants, test_variants = arguments
    
    # Change regression labels to binary labels above first quartile and below
    original_y_train = y_train.copy()
    original_y_test = y_test.copy()
    if is_classifier(model):
        y_train = np.where(y_train >= np.percentile(y_train, 75), 1, 0).ravel()
        y_test = np.where(y_test >= np.percentile(y_test, 75), 1, 0).ravel()

    if enc2 == None:
        ss_results_file = os.path.join(results_folder, f'pred_dict_ss_{enc1}_{labeled_percentage}.pkl')
    else:
        ss_results_file = os.path.join(results_folder, f'pred_dict_ss_{enc1}_{enc2}_{labeled_percentage}.pkl')

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
    pred_dicts_ct = []
    metadata_dict = {"Encoding1": enc1,
                     "Encoding2": enc2,
                     "Labeled percentage": labeled_percentage,
                     "Model": model.__class__.__name__,
                     "SS Method": ss_method.__class__.__name__,
                     "normalization_method": normalization_method,
                     "n_splits": n_splits,
                     "preds_file": ss_results_file,
                     "pred_mode": pred_mode,
                     "Train variants": train_variants,
                     "Test variants": test_variants} 
    
    # We can't use a classic skf split because the train and test are defined by the number of variants
    # So we just run it n_splits times
    arguments = []
    for k in range(n_splits):
        temp_results_file = os.path.join(results_folder, f'pred_dict_ss_{enc1}_{enc2}_{labeled_percentage}_{k}.pkl')
        #fold([model, enc1, enc2, enc1_X_train, enc2_X_train, enc1_X_test, enc2_X_test, y_train, y_test, labeled_percentage, ss_method, pred_dicts_ct, k, original_y_test])
        arguments.append([model, enc1, enc2, enc1_X_train, enc2_X_train, enc1_X_test, enc2_X_test, y_train, y_test, labeled_percentage, ss_method, pred_dicts_ct, k, original_y_test, temp_results_file])
        
    # Create k threads that call fold function
    with Pool(n_splits) as pool:
        pool.map(fold, arguments)
        pool.close()
        pool.join()
    
    # Read temp files and merge them
    for k in range(n_splits):
        temp_results_file = os.path.join(results_folder, f'pred_dict_ss_{enc1}_{enc2}_{labeled_percentage}_{k}.pkl')
        with open(temp_results_file, 'rb') as handle:
            pred_dicts_ct.extend(pkl.load(handle))
        os.remove(temp_results_file)
    

    # Save dicts to pickle files
    with open(ss_results_file, 'wb') as handle:
        pkl.dump(pred_dicts_ct, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # Metadata
    with open(ss_results_file.replace(".pkl", "_metadata.pkl"), 'wb') as handle:
        pkl.dump(metadata_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


    # Free memory (it seems threads are waiting taking memory innecesarily)
    del enc1_X_train
    del enc1_X_test
    
    if enc2:
        del enc2_X_train
        del enc2_X_test
    
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
        "--ssmethod", 
        type=str,
        default="SelfTraining"
    )
    CLI.add_argument(
        "--basemodel", 
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
        "--enc1",
        type=str,
        default="One_hot"
    )
    CLI.add_argument(
        "--enc2",
        type=str,
        default=None
    )
    CLI.add_argument(
        "--labeled",
        nargs="*",
        type=float,
        default=[0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.0025, 0.001]
    )

    
    dataset_folder = CLI.parse_args().data
    dataset = dataset_folder.split('data/')[-1].split('/')[0]
    
    # Create model
    if CLI.parse_args().basemodel.lower() == "decisiontreeregressor":
        model = DecisionTreeRegressor()
    elif CLI.parse_args().basemodel.lower() == "ridge":
        model = Ridge(alpha=0.85)
    elif CLI.parse_args().basemodel.lower() == "ridgeclassifier":
        model = RidgeClassifier()
    elif CLI.parse_args().basemodel.lower() == "logisticregression":
        model = LogisticRegression(max_iter=5000)
    elif CLI.parse_args().basemodel.lower() == "svr":
        model = SVR()
    elif CLI.parse_args().basemodel.lower() == "multiviewcoregression":
        model = MultiviewCoReg()
    else:
        raise ValueError("Model not supported")
    
    # SS method
    if CLI.parse_args().ssmethod.lower() == "selftraining":
        ss_method = SelfTraining
        print("SSmethod: ", ss_method.__class__.__name__)
    elif CLI.parse_args().ssmethod.lower() == "tritrainingregressor":
        ss_method = TriTrainingRegressor(base_estimator=clone(model), y_tol_per=1)
        print("SSmethod: ", ss_method.__class__.__name__)
    elif CLI.parse_args().ssmethod.lower() == "multiviewcoregression":
        ss_method = MultiviewCoReg(max_iters=100, pool_size=100, p1=2, p2=5)
        print("SSmethod: ", ss_method.__class__.__name__)
    else:
        raise ValueError("SS method not supported")

    if CLI.parse_args().normalize == "True":
        model = make_pipeline(StandardScaler(), model)
    
    if hasattr(model, 'steps'):
        model_name = '_'.join([submodel.__class__.__name__ for submodel in model])
    else:
        model_name = model.__class__.__name__

    # Create results folder
    if CLI.parse_args().enc2:
        experiments_id = f"semisupervised_extrapolation_experiments_{dataset}_{model_name}_{CLI.parse_args().ssmethod}_{CLI.parse_args().enc1}_{CLI.parse_args().enc2}"
    else:
        # experiments_id = f"semisupervised_extrapolation_experiments_{dataset}_{model_name}_{CLI.parse_args().ssmethod}_{CLI.parse_args().enc1}"
        experiments_id = f"semisupervised_extrapolation_experiments_{dataset}_{model_name}_{CLI.parse_args().ssmethod}"
    
    labeled_percentages = CLI.parse_args().labeled
    
    y_file = os.path.join(dataset_folder, dataset+"_y.pkl")

    y = pkl.load(open(y_file, 'rb'))

    encoding_name = CLI.parse_args().enc1

    start = time.time()
    encoding_file = os.path.join(dataset_folder, f'X_{encoding_name}.pkl')
    if CLI.parse_args().enc2:
        enc2 = CLI.parse_args().enc2
        encoding_file2 = os.path.join(dataset_folder, f'X_{enc2}.pkl')
    else:
        enc2 = None
        encoding_file2 = None
                
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
        X_indexes = []
        for variant in train_variants:
            X_indexes.extend(variants_dict[variant])

        train_indexes = random.sample(X_indexes, int(len(X_indexes)*0.8))
        test_indexes = list(set(X_indexes) - set(train_indexes))

        # Only use a 10% of the test indexes (knn is very slow predicting)
        test_indexes = np.random.choice(test_indexes, int(len(test_indexes)*0.1), replace=False)

    experiments_id = experiments_id.replace("experiments_", f"experiments_trainedwith_{'_'.join(str(x) for x in train_variants)}_testedwith_{'_'.join(str(x) for x in test_variants)}_")
    results_folder = os.path.join("results", experiments_id)

    # X
    # If file does not exist, encode
    if not os.path.exists(encoding_file):
        print(f"--> Encoding {encoding_name}...", end="", flush=True)
        enc_X = np.array([SequenceEncoding(encoding_name).get_encoding(seq) for seq in X])
        enc_X = enc_X.reshape(enc_X.shape[0], -1)
        # Save the encoding to a pickle file
        with open(encoding_file, 'wb') as handle:
            pkl.dump(enc_X, handle, protocol=pkl.HIGHEST_PROTOCOL) 
        
        if enc2:
            enc2_X = np.array([SequenceEncoding(enc2).get_encoding(seq) for seq in X])
            enc2_X = enc2_X.reshape(enc2_X.shape[0], -1)
            # Save the encoding to a pickle file
            with open(encoding_file2, 'wb') as handle:
                pkl.dump(enc2_X, handle, protocol=pkl.HIGHEST_PROTOCOL)       
        else:
            enc2_X = None
        
        print(f"\tsize={round(enc_X.nbytes/(1024*1024), 2)} MB\t time={time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}", flush=True)
    else:
        enc_X = pkl.load(open(encoding_file, 'rb'))
        if enc2:
            enc2_X = pkl.load(open(encoding_file2, 'rb'))
        else:
            enc2_X = None

    arguments = []
    for labeled_percentage in labeled_percentages:
        if enc2:
            arguments.append([encoding_name, enc2,
                            enc_X[train_indexes],  # enc1_X_train
                            enc2_X[train_indexes], # enc2_X_train
                            enc_X[test_indexes],   # enc_X_test
                            enc2_X[test_indexes],  # enc2_X_test
                            y[train_indexes],      # y_train
                            y[test_indexes],       # y_test
                            labeled_percentage,
                            ss_method,
                            model,
                            results_folder,
                            train_variants,
                            test_variants])
        else:
            arguments.append([encoding_name, None,
                            enc_X[train_indexes],  # enc1_X_train
                            None,                  # enc2_X_train
                            enc_X[test_indexes],   # enc_X_test
                            None,                  # enc2_X_test
                            y[train_indexes],      # y_train
                            y[test_indexes],       # y_test
                            labeled_percentage,
                            ss_method,
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
    # n_cores = CLI.parse_args().cpus
    # with Pool(n_cores) as pool:
    #     pool.map(main, arguments, chunksize=1)
    
    for arg in arguments:
        main(arg)