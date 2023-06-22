# Include sslearn folder
import sys
import os
import warnings
import pandas as pd
import gc
import pickle as pkl
import time
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len, InputEncoder
from proteinbert.finetuning import encode_dataset
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

import mkl
mkl.set_num_threads(1)
import argparse

def proteinbert_finetuning(X_train, y_train, X_val, y_val, seq_len, pred_mode):
    if pred_mode == "regression":    
        OUTPUT_TYPE = OutputType(False, 'numeric')
        OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE)
    elif pred_mode == "classification":
        OUTPUT_TYPE = OutputType(False, 'binary')
        UNIQUE_LABELS = [0, 1]  
        OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

    # Best parameters after grid search
    batch_size = 8
    max_epochs_per_stage = 1000
    lr = 1e-04
    lr_with_frozen_pretrained_layers = 1e-04
    final_lr = 1e-05
    n_final_epochs = 200
    final_seq_len = seq_len+2
    dropout_rate = 0.5
    min_lr = 1e-09
    factor = 0.1
    patience_red = 50
    patience_early = 60
    begin_with_frozen_pretrained_layers = True

    # Loading the pre-trained model and fine-tuning it on the loaded dataset
    pretrained_model_generator, input_encoder = load_pretrained_model()

    # get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
    model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
                                            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs, dropout_rate=dropout_rate)

    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience=patience_red, factor=factor, min_lr=min_lr, verbose=1),
        keras.callbacks.EarlyStopping(patience=patience_early, restore_best_weights=True),
    ]

    finetune(model_generator, input_encoder, OUTPUT_SPEC, 
             X_train, y_train, X_val, y_val, seq_len=seq_len+2, 
             batch_size=batch_size, max_epochs_per_stage=max_epochs_per_stage, lr=lr,                                     
             begin_with_frozen_pretrained_layers=begin_with_frozen_pretrained_layers, 
             lr_with_frozen_pretrained_layers=lr_with_frozen_pretrained_layers, 
             n_final_epochs=n_final_epochs, final_seq_len=final_seq_len, 
             final_lr=final_lr, callbacks=training_callbacks)
    return model_generator.create_model(seq_len+2), input_encoder, OUTPUT_SPEC

def main(X_train, X_val, X_test, y_train, y_val, y_test, labeled_percentage, pred_mode, results_folder, train_variants, test_variants):
    
    # Change regression labels to binary labels above first quartile and below
    original_X_test = X_test.copy()
    original_y_test = y_test.copy()

    if pred_mode == "classification":
        y_train = np.where(y_train >= np.percentile(y_train, 75), 1, 0).ravel()
        y_val = np.where(y_val >= np.percentile(y_val, 75), 1, 0).ravel()
        y_test = np.where(y_test >= np.percentile(y_test, 75), 1, 0).ravel()
        
    elif pred_mode == "regression":
        y_train = y_train.ravel()
        y_val = y_val.ravel()
        y_test = y_test.ravel()
    else:
        raise ValueError(f"pred_mode must be classification or regression, not {pred_mode}")

    results_file = os.path.join(results_folder, f'pred_dict_proteinbert_{labeled_percentage}.pkl')

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    # Stratified kfold
    n_splits = 5

    # Initialize dicts to store results
    pred_dicts = []
    metadata_dict = {"Encoding": "None",
                     "Labeled percentage": labeled_percentage,
                     "Model": "ProteinBERT",
                     "SS Method": "None",
                     "normalization_method": "None",
                     "n_splits": n_splits,
                     "preds_file": results_file,
                     "pred_mode": pred_mode,
                     "Train variants": train_variants,
                     "Test variants": test_variants}
    
    # We can't use a classic skf split because the train and test are defined by the number of variants
    # So we just run it n_splits times
    for k in range(n_splits):

        start = time.time()

        if pred_mode == "classification":
            # Get a stratifed sample with size = labeled_percentage
            # This is a bit strange. I am using StratifiedShuffleSplit to get labeled/unlabeled indexes
            # from enc_X_train and y_train.
            # Then I am using those indexes to get the corresponding instances.
            # "test_size == labeled_percentage"
            if labeled_percentage < 1:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=labeled_percentage)
                unlabeled_indexes, labeled_indexes = next(sss.split(X_train, y_train))
            # If labeled_percentage == 1, then we don't need to stratify
            elif labeled_percentage == 1:
                labeled_indexes = np.arange(len(X_train))
                unlabeled_indexes = []
        else:
            if labeled_percentage < 1:
                # Get labeled_percentage random indexed from enc_X_train
                labeled_indexes = np.random.choice(len(X_train), int(labeled_percentage * len(X_train)), replace=False)
                unlabeled_indexes = np.setdiff1d(np.arange(len(X_train)), labeled_indexes)
            elif labeled_percentage == 1:
                labeled_indexes = np.arange(len(X_train))
                unlabeled_indexes = []

        # Get unlabeled subsets
        X_train_onlylabeled = np.delete(X_train, unlabeled_indexes, axis=0)
        y_train_onlylabeled = np.delete(y_train, unlabeled_indexes, axis=0)
        
        # Model
        # If results file already exists, don't run model
        model, input_encoder, output_spec = proteinbert_finetuning(X_train_onlylabeled, y_train_onlylabeled, X_val, y_val, seq_len, pred_mode)
        # Turn X_test into a list of strings
        dataset = pd.DataFrame({'seq': original_X_test, 'raw_y': original_y_test})
        original_X_test = dataset['seq']
        original_y_test = dataset['raw_y']
        X_test, original_y_test, sample_weights = encode_dataset(original_X_test, original_y_test, input_encoder, output_spec, seq_len = seq_len, needs_filtering = False, \
            dataset_name = 'Test set')

        model_y_proba = model.predict(X_test, batch_size=32).ravel()
        
        pred_dicts.append({"y_proba": model_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})
        
        # Print formatted taken time in hours, minutes and seconds
        print(f"\tExperiment with proteinbert using {labeled_percentage*100}% labeled instances (k={k}) took {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")

    with open(results_file, 'wb') as handle:
        pkl.dump(pred_dicts, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # Metadata
    with open(results_file.replace(".pkl", "_metadata.pkl"), 'wb') as handle:
        pkl.dump(metadata_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


    # Free memory (it seems threads are waiting taking memory innecesarily)
    del X_train
    del X_test
    del X_train_onlylabeled
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
        "--predmode",
        type=str
    )
    CLI.add_argument(
        "--labeled",
        nargs="*",
        type=float,
        default=[1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.0025, 0.001]
    )
    
    dataset_folder = CLI.parse_args().data
    dataset = dataset_folder.split('data/')[-1].split('/')[0]

    # Create results folder
    experiments_id = f"proteinbert_extrapolation_experiments_{dataset}_proteinbert_{CLI.parse_args().predmode}"
    
    labeled_percentages = CLI.parse_args().labeled
    
    y_file = os.path.join(dataset_folder, dataset+"_y.pkl")

    y = pkl.load(open(y_file, 'rb'))

    start = time.time()
    
    # If file does not exist, encode
    X_file = os.path.join(dataset_folder, dataset+"_X.pkl")
    X = pkl.load(open(X_file, 'rb'))
    
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

    # Getting the length of the sequences
    seq_len = len(X[0])

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

        # Train/test by variant number
        X_train = np.array([X[i] for i in train_indexes])
        y_train = np.array([y[i] for i in train_indexes])
        X_test = np.array([X[i] for i in test_indexes])
        y_test = np.array([y[i] for i in test_indexes])

        # Val set from train set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42)
        

    else:
        X_indexes = []
        for variant in train_variants:
            X_indexes.extend(variants_dict[variant])

        X = np.array([X[i] for i in X_indexes])
        y = np.array([y[i] for i in X_indexes])
        
        # Train/val/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42)
        

    
    print("X_train shape:", X_train.shape)
    print("X_train type:", type(X_train))
    print("y_train shape:", y_train.shape)
    print("y_train type:", type(y_train))
    print("X_val shape:", X_val.shape)
    print("X_val type:", type(X_val))
    print("y_val shape:", y_val.shape)
    print("y_val type:", type(y_val))
    print("X_test shape:", X_test.shape)
    print("X_test type:", type(X_test))
    print("y_test shape:", y_test.shape)
    print("y_test type:", type(y_test))

    experiments_id = experiments_id.replace("experiments_", f"experiments_trainedwith_{'_'.join(str(x) for x in train_variants)}_testedwith_{'_'.join(str(x) for x in test_variants)}_")
    results_folder = os.path.join("results", experiments_id)

    for labeled_percentage in labeled_percentages:
        main(X_train,X_val,X_test,
             y_train,y_val,y_test,
             labeled_percentage,
             CLI.parse_args().predmode,
             results_folder,
             train_variants,
             test_variants)
