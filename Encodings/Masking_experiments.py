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

from Bio import SeqIO
import matplotlib.pyplot as plt
import numpy as np
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
from scipy.ndimage import gaussian_filter
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
                                                
def get_variant_position_mask(wt_seq_encoded, X, weight=2, apply_gaussian_filter=False, gaussian_filter_sigma=1):
    # Creates a mask that masks all positions that are not the same as the wildtype sequence
    mask = np.ones(X.shape)
    for i, seq in enumerate(X):
        mask[i, np.where(seq != wt_seq_encoded)[0], :] = weight
    if apply_gaussian_filter:
        for i, seq in enumerate(mask):
            mask[i, :, :] = gaussian_filter(seq, sigma=gaussian_filter_sigma)

    return mask

def main(enc, enc_X, global_masks, individual_masks, wt_seq, y, labeled_percentage, model, results_folder):
    
    original_y = y.copy()
    
    pred_dict = dict()
    pred_dict['unmasked'] = dict()
    for mask_name, mask in global_masks.items():
        pred_dict[mask_name] = dict()
    for mask_name, mask in individual_masks.items():
        pred_dict[mask_name] = dict()

    results_files_dict = dict()
    results_files_dict['unmasked'] = os.path.join(results_folder, f'pred_dict_{enc}_{labeled_percentage}.pickle')
    for mask_name, mask in global_masks.items():
        results_files_dict[mask_name] = os.path.join(results_folder, f'pred_dict_{enc}_masked_{mask_name}_{labeled_percentage}.pickle')
    for mask_name, mask in individual_masks.items():
        results_files_dict[mask_name] = os.path.join(results_folder, f'pred_dict_{enc}_masked_{mask_name}_{labeled_percentage}.pickle')


    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    
    k = 0
    
    split_func = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in split_func.split(enc_X, y):
        
        k += 1
        
        enc_X_train, enc_X_test = enc_X[train_index], enc_X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        original_y_train, original_y_test = original_y[train_index], original_y[test_index]

        start = time.time()

        # Get labeled_percentage random indexed from enc1_X_train
        if labeled_percentage < 1:
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
        for mask_name, mask in global_masks.items():
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
        
        # Model with individual masks
        for mask_name, args in individual_masks.items():
            get_mask = args[0]
            weight = args[1]
            if len(args) > 2:
                apply_gaussian_filter = args[2]
            else:
                apply_gaussian_filter = False
            masked_model = clone(model)
            masked_X_train = enc_X_train_onlylabeled * get_mask(wt_seq, enc_X_train_onlylabeled, weight=weight, gaussian_filter=apply_gaussian_filter)
            masked_X_train = masked_X_train.reshape(masked_X_train.shape[0], -1) # Flatten
            masked_X_test = enc_X_test * get_mask(wt_seq, enc_X_test, weight=weight)
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
    del enc_X
    del enc_X_train
    del enc_X_test
    del enc_X_train_onlylabeled
    del y_train
    del y_train_ct
    del y_train_onlylabeled
    del y_test
    gc.collect()


if __name__ == "__main__":

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
    
    dataset_folder = CLI.parse_args().data
    dataset = dataset_folder.split('data/')[-1].split('/')[0]
    
    model = Ridge()
    results_folder = f"results/masking_experiments_{dataset}_{model.__class__.__name__}/"

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
    wt_seq = SeqIO.read(wt_seq_file, "fasta").seq
    starting_pos = get_wt_starting_position(wt_seq_file)
    relative_entropy_mask = get_sequence_conservation_mask(msa_file, method="relative")
    shannon_entropy_mask = get_sequence_conservation_mask(msa_file, method="shannon")
    lockless_entropy_mask = get_sequence_conservation_mask(msa_file, method="lockless")

    global_masks = {"relative": relative_entropy_mask,
                    "shannon": shannon_entropy_mask,
                    "lockless": lockless_entropy_mask,
                    "1-shannon": 1 - shannon_entropy_mask,
                    "inverted_relative": (1 / 1 + relative_entropy_mask),
                    "inverted_shannon": (1 / 1 + shannon_entropy_mask),
                    "inverted_lockless": (1 / 1 + lockless_entropy_mask),
                    "normalized_relative": relative_entropy_mask / np.mean(relative_entropy_mask),
                    "normalized_shannon": shannon_entropy_mask / np.mean(shannon_entropy_mask),
                    "normalized_lockless": lockless_entropy_mask / np.mean(lockless_entropy_mask),
                    "random": np.random.rand(relative_entropy_mask.shape[0], relative_entropy_mask.shape[1], 1),
                    }
    individual_masks = {"variants_emphasis_weight_2": (get_variant_position_mask, 2),
                        "variants_emphasis_weight_5": (get_variant_position_mask, 5),
                        "variants_emphasis_weight_10": (get_variant_position_mask, 10),
                        "variants_emphasis_weight_100": (get_variant_position_mask, 100),
                        "variants_gaussian_emphasis_weight_2": (get_variant_position_mask, 2, True),
                        "variants_gaussian_emphasis_weight_5": (get_variant_position_mask, 5, True),
                        "variants_gaussian_emphasis_weight_10": (get_variant_position_mask, 10, True),
                        "variants_gaussian_emphasis_weight_100": (get_variant_position_mask, 100, True)}
    encodings_dict = dict()
    wt_encodings_dict = dict()
    i=0
    for encoding_name in encoding_names:
        start = time.time()
        encoding_file = os.path.join(dataset_folder, f'X_{encoding_name}.pkl')
        encoding_wt_file =  os.path.join(dataset_folder, f'wt_{encoding_name}.pkl')
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
        
        if not os.path.exists(encoding_wt_file):
            # Encoded wt
            enc_wt = SequenceEncoding(encoding_name).get_encoding(wt_seq)
            with open(encoding_wt_file, 'wb') as handle:
                pkl.dump(enc_wt, handle, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            enc_wt = pkl.load(open(encoding_wt_file, 'rb'))
        
        encodings_dict[encoding_name] = enc_X
        wt_encodings_dict[encoding_name] = enc_wt
    
    print(f"* Total dict size: {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024), 2)} MB | {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024*1024), 2)} GB", flush=True)
    arguments = []
    for labeled_percentage in labeled_percentages:
        arguments.extend([(enc, encodings_dict[enc], global_masks, individual_masks, wt_encodings_dict[enc], y.copy(), labeled_percentage, clone(model), results_folder) for enc in encoding_names])
    print(f"* Total number of experiments: {len(arguments)}")
    print(f"* Number of cores: {CLI.parse_args().cpus}")
    print(f"* Starting experiments...")

    # To avoid unintented multithreading:
    # https://stackoverflow.com/questions/19257070/unintended-multithreading-in-python-scikit-learn/42124978#42124978
    # terminal: export OPENBLAS_NUM_THREADS=1
    # To know numpy/scipy config: https://stackoverflow.com/questions/9000164/how-to-check-blas-lapack-linkage-in-numpy-and-scipy
    n_cores = CLI.parse_args().cpus
    with Pool(n_cores) as pool:
        pool.starmap(main, arguments, chunksize=1)
