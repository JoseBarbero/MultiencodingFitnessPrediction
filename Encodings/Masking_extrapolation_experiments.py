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
from Bio import SeqIO
from SequenceEncoding import SequenceEncoding
from sklearn.base import TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, StackingRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sslearn.base import OneVsRestSSLClassifier
from sslearn.wrapper import CoTraining
from sklearn.base import is_classifier, is_regressor
from pycanal import Canal
import mkl
from scipy.ndimage import gaussian_filter1d
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

def get_sequence_conservation_mask(msa_file, method, ref_pos=0, startcount=0):

    # Create an instance of the Canal class
    canal = Canal(fastafile=msa_file, #Multiple sequence alignment (MSA) of homologous sequences
                ref=ref_pos, #Position of reference sequence in MSA, use first sequence
                startcount=startcount, #Position label of first residue in reference sequence
                verbose=True # Print out progress
                )
    return canal.analysis(include=None, method=method)[method].values.reshape(1, -1, 1)

def get_wt_starting_position(wt_seq_file):
    # Returns the starting position of the wildtype sequence in the MSA
    wt_seq = open(wt_seq_file, 'r')
    # Get the first line
    wt_seq_header = wt_seq.readline()
    if "start:" in wt_seq_header:
        starting_pos = int(wt_seq_header.split('start:')[1].split(',')[0].strip())
    else:
        starting_pos = 0
    return starting_pos
                                                
def get_variant_position_mask(wt_seq_file, X,):
    # Creates a mask that masks all positions that are not the same as the wildtype sequence
    wt_seq = open(wt_seq_file, 'r').read().strip()
    mask = np.zeros(X.shape)
    for i, seq in enumerate(X):
        mask[i, np.where(seq != wt_seq)[0], 0] = 1
    return mask

def main(enc, enc_X_train, enc_X_test, masks, y_train, y_test, labeled_percentage, model, results_folder):
    
    original_y_train = y_train.copy()
    original_y_test = y_test.copy()
    
    pred_dict = dict()
    pred_dict['unmasked'] = dict()
    for mask_name, mask in masks.items():
        pred_dict[mask_name] = dict()

    results_files_dict = dict()
    results_files_dict['unmasked'] = os.path.join(results_folder, f'pred_dict_{enc}_{labeled_percentage}.pickle')
    for mask_name, mask in masks.items():
        results_files_dict[mask_name] = os.path.join(results_folder, f'pred_dict_{enc}_masked_{mask_name}_{labeled_percentage}.pickle')

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    n_splits = 5
    
    # We can't use a classic skf split because the train and test are defined by the number of variants
    # So we just run it n_splits times
    for k in range(n_splits):

        start = time.time()

        if labeled_percentage < 1:
            # Get labeled_percentage random indexed from enc1_X_train
            labeled_indexes = np.random.choice(len(enc_X_train), int(labeled_percentage * len(enc_X_train)), replace=False)
            unlabeled_indexes = np.setdiff1d(np.arange(len(enc_X_train)), labeled_indexes)
        elif labeled_percentage == 1:
            labeled_indexes = np.arange(len(enc_X_train))
            unlabeled_indexes = []

        # Get unlabeled subsets
        enc_X_train_onlylabeled = np.delete(enc_X_train, unlabeled_indexes, axis=0)
        y_train_onlylabeled = np.delete(y_train, unlabeled_indexes, axis=0)
        
        # Set unlabeled_indexes to -1 in y
        y_train_ct = np.copy(y_train)
        y_train_ct[unlabeled_indexes] = -1

        # Model unmasked
        if not os.path.exists(results_files_dict['unmasked']):
            unmasked_model = clone(model)
            unmasked_X_train = enc_X_train_onlylabeled.reshape(enc_X_train_onlylabeled.shape[0], -1) # Flatten
            unmasked_X_test = enc_X_test.reshape(enc_X_test.shape[0], -1) # Flatten
            unmasked_model.fit(unmasked_X_train, y_train_onlylabeled)
            unmasked_model_y_proba = unmasked_model.predict(unmasked_X_test)
            pred_dict['unmasked'][k] = {"y_proba": unmasked_model_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)}

        # Model masked
        for mask_name, mask in masks.items():
            if not os.path.exists(results_files_dict[mask_name]):
                masked_model = clone(model)
                # We mask X_train and X_test by multiplying sequences in X by the mask
                masked_X_train = enc_X_train_onlylabeled * mask
                masked_X_train = masked_X_train.reshape(masked_X_train.shape[0], -1) # Flatten
                masked_X_test = enc_X_test * mask
                masked_X_test = masked_X_test.reshape(masked_X_test.shape[0], -1) # Flatten
                masked_model.fit(masked_X_train, y_train_onlylabeled)
                masked_model_y_proba = masked_model.predict(masked_X_test)
                pred_dict[mask_name][k] = {"y_proba": masked_model_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)}
            
        # Print formatted taken time in hours, minutes and seconds
        print(f"\tExperiment with {enc} using {labeled_percentage*100}% labeled instances (k={k}) took {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")

    # Save dicts to pickle files
    for method, results_file in results_files_dict.items():
        if not os.path.exists(results_file):
            with open(results_file, 'wb') as handle:
                pkl.dump(pred_dict[method], handle, protocol=pkl.HIGHEST_PROTOCOL)

    # Free memory (it seems threads are waiting taking memory innecesarily)
    del enc_X_train
    del enc_X_test
    del enc_X_train_onlylabeled
    del y_train
    del y_train_ct
    del y_train_onlylabeled
    del y_test
    gc.collect()


if __name__ == "__main__":

    # How to run:
    # python Masking_extrapolation_experiments.py data/iris/ --cpus 32 --trainvariants 1 2 --testvariants 3
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
    
    model = Ridge()
    results_folder = f"results/masking_extrapolation_experiments_{dataset}_{model.__class__.__name__}/"

    labeled_percentages = [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.03, 0.01]
    
    y_file = os.path.join(dataset_folder, dataset+"_y.pkl")
    y = pkl.load(open(y_file, 'rb'))
    # If y is a list, convert it to a numpy array
    if isinstance(y, list):
        y = np.array(y)
    
    encoding_names = ["One_hot", "One_hot_6_bit", "Binary_5_bit",
                      "Hydrophobicity_matrix", "Meiler_parameters", "Acthely_factors",
                      "PAM250", "BLOSUM62",
                      "Miyazawa_energies", "Micheletti_potentials",
                      "AESNN3", "ANN4D", 
                    #   "ProtVec"   # ProtVec is problematic because it uses different lenght (233 instead of 235 because it works by 3-mers)
                      ]
    
    msa_file = os.path.join(dataset_folder, "aligned_seqs.fasta")
    wt_seq_file = os.path.join(dataset_folder, dataset+"_wt.fasta")
    starting_pos = get_wt_starting_position(wt_seq_file)
    relative_entropy_mask = get_sequence_conservation_mask(msa_file, method="relative")
    shannon_entropy_mask = get_sequence_conservation_mask(msa_file, method="shannon")
    lockless_entropy_mask = get_sequence_conservation_mask(msa_file, method="lockless")

    masks = {"relative": relative_entropy_mask,
             "shannon": shannon_entropy_mask,
             "lockless": lockless_entropy_mask,
             "inverted_relative": (1 / 1 + relative_entropy_mask),
             "inverted_shannon": (1 / 1 + shannon_entropy_mask),
             "inverted_lockless": (1 / 1 + lockless_entropy_mask),
             "normalized_relative": relative_entropy_mask / np.max(relative_entropy_mask),
             "normalized_shannon": shannon_entropy_mask / np.max(shannon_entropy_mask),
             "normalized_lockless": lockless_entropy_mask / np.max(lockless_entropy_mask),
             "random": np.random.rand(relative_entropy_mask.shape[0], relative_entropy_mask.shape[1], 1)
             }
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
            #enc_X = enc_X.reshape(enc_X.shape[0], -1)
            # Save the encoding to a pickle file
            with open(encoding_file, 'wb') as handle:
                pkl.dump(enc_X, handle, protocol=pkl.HIGHEST_PROTOCOL) 
            print(f"\tsize={round(enc_X.nbytes/(1024*1024), 2)} MB\t time={time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}", flush=True)
        else:
            enc_X = pkl.load(open(encoding_file, 'rb'))
        
        encodings_dict[encoding_name] = enc_X

    wt_sequence = SeqIO.read(wt_seq_file, "fasta").seq
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

    train_indexes = []
    for variant in train_variants:
        train_indexes.extend(variants_dict[variant])
    test_indexes = []
    for variant in test_variants:
        test_indexes.extend(variants_dict[variant])

    results_folder = results_folder.replace("experiments_", f"experiments_trainedwith_{'_'.join(str(x) for x in train_variants)}_testedwith_{'_'.join(str(x) for x in test_variants)}_")
    
    print(f"* Total dict size: {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024), 2)} MB | {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024*1024), 2)} GB", flush=True)
    arguments = []
    for labeled_percentage in labeled_percentages:
        arguments.extend([( enc,
                            encodings_dict[enc][train_indexes],
                            encodings_dict[enc][test_indexes],
                            masks,
                            y[train_indexes],
                            y[test_indexes],
                            labeled_percentage,
                            model,
                            results_folder) for enc in encoding_names])
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
        pool.starmap(main, arguments, chunksize=1)