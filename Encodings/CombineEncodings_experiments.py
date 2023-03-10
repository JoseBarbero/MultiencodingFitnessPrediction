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
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, StackingRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sslearn.base import OneVsRestSSLClassifier
from sslearn.wrapper import CoTraining
from sklearn.base import is_classifier, is_regressor

import mkl
mkl.set_num_threads(1)

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

def main(enc1, enc2, enc1_X, enc2_X, y, labeled_percentage, model, results_folder):
    
    # Change regression labels to binary labels above first quartile and below
    original_y = y.copy()
    y = np.where(y >= np.percentile(y, 75), 1, 0).ravel()

    pred_dict_ct = dict()
    pred_dict_enc1 = dict()
    pred_dict_enc2 = dict()
    pred_dict_concat = dict()
    pred_dict_st_lr = dict()
    pred_dict_st_mean = dict()

    ct_results_file = os.path.join(results_folder, f'pred_dict_ct_{enc1}_{enc2}_{labeled_percentage}.pickle')
    enc1_results_file = os.path.join(results_folder, f'pred_dict_{enc1}_{labeled_percentage}.pickle')
    enc2_results_file = os.path.join(results_folder, f'pred_dict_{enc2}_{labeled_percentage}.pickle')
    concat_results_file = os.path.join(results_folder, f'pred_dict_concat_{enc1}_{enc2}_{labeled_percentage}.pickle')
    st_lr_results_file = os.path.join(results_folder, f'pred_dict_st_lr_{enc1}_{enc2}_{labeled_percentage}.pickle')
    st_mean_results_file = os.path.join(results_folder, f'pred_dict_st_mean_{enc1}_{enc2}_{labeled_percentage}.pickle')

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # Touch files if they don't exist so the other processes don't try to run them
    # Enc1 and Enc2 only need to be run one (but every thread could run them)
    # save_enc1/2 indicates if this thread should run the models and save the results
    run_enc1 = False
    run_enc2 = False
    if not os.path.exists(enc1_results_file):
        Path(enc1_results_file).touch()
        run_enc1 = True
    if not os.path.exists(enc2_results_file):
        Path(enc2_results_file).touch()
        run_enc2 = True
        
    # Stratified kfold
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    # Initialize dicts to store results
    pred_dict_ct = []
    pred_dict_enc1 = []
    pred_dict_enc2 = []
    pred_dict_concat = []
    pred_dict_st_lr = []
    pred_dict_st_mean = []
    
    k = 0
    for train_index, test_index in skf.split(enc1_X, y):
        k += 1

        enc1_X_train, enc1_X_test = enc1_X[train_index], enc1_X[test_index]
        enc2_X_train, enc2_X_test = enc2_X[train_index], enc2_X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        original_y_train, original_y_test = original_y[train_index], original_y[test_index]

        start = time.time()

        # Get a stratifed sample with size = labeled_percentage
        # This is a bit strange. I am using StratifiedShuffleSplit to get labeled/unlabeled indexes
        # from enc1_X_train and y_train. 
        # Then I am using those indexes to get the corresponding instances.
        # "test_size == labeled_percentage"
        sss = StratifiedShuffleSplit(n_splits=1, test_size=labeled_percentage)
        unlabeled_indexes, labeled_indexes = next(sss.split(enc1_X_train, y_train))

        # Get unlabeled subsets
        enc1_X_train_onlylabeled = np.delete(enc1_X_train, unlabeled_indexes, axis=0)
        enc2_X_train_onlylabeled = np.delete(enc2_X_train, unlabeled_indexes, axis=0)
        y_train_onlylabeled = np.delete(y_train, unlabeled_indexes, axis=0)
        
        # Set unlabeled_indexes to -1 in y
        y_train_ct = np.copy(y_train)
        y_train_ct[unlabeled_indexes] = -1

        # Get concatenated X
        concat_X_train_onlylabeled = np.concatenate((enc1_X_train_onlylabeled, enc2_X_train_onlylabeled), axis=1)
        concat_X_test = np.concatenate((enc1_X_test, enc2_X_test), axis=1)

        # Co-training model
        if is_classifier(model):
            if not os.path.exists(ct_results_file):
                ct = CoTraining(base_estimator=clone(model))
                ct.fit(enc1_X_train, y_train_ct, X2=enc2_X_train)
                ct_y_proba = ct.predict_proba(enc1_X_test, X2=enc2_X_test)[:, 1]
                pred_dict_ct.append({"y_proba": ct_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})

        # Model enc1
        # If results file already exists, don't run model
        if run_enc1:
            enc1_model = clone(model)
            enc1_model.fit(enc1_X_train_onlylabeled, y_train_onlylabeled)
            if is_classifier(model):
                enc1_model_y_proba = enc1_model.predict_proba(enc1_X_test)[:, 1]
            elif is_regressor(model):
                enc1_model_y_proba = enc1_model.predict(enc1_X_test)
            else:
                raise ValueError('Model must be a classifier or regressor')
            pred_dict_enc1.append({"y_proba": enc1_model_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})
            
        # Model enc2
        if run_enc2:
            enc2_model = clone(model)
            enc2_model.fit(enc2_X_train_onlylabeled, y_train_onlylabeled)
            if is_classifier(model):
                enc2_model_y_proba = enc2_model.predict_proba(enc2_X_test)[:, 1]
            elif is_regressor(model):
                enc2_model_y_proba = enc2_model.predict(enc2_X_test)
            else:
                raise ValueError('Model must be a classifier or regressor')
            pred_dict_enc2.append({"y_proba": enc2_model_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})
        
        # Model concatenating enc1+enc2
        if not os.path.exists(concat_results_file):
            concat_model = clone(model)
            concat_model.fit(concat_X_train_onlylabeled, y_train_onlylabeled)
            if is_classifier(model):
                concat_model_y_proba = concat_model.predict_proba(concat_X_test)[:, 1]
            elif is_regressor(model):
                concat_model_y_proba = concat_model.predict(concat_X_test)
            pred_dict_concat.append({"y_proba": concat_model_y_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})

        # Model stacking enc1+enc2
        if not os.path.exists(st_lr_results_file):
            stacking_estimators = [('enc1',
                                    make_pipeline(ColumnExtractor(range(0, enc1_X_train.shape[1])), 
                                                    clone(model))),
                                    ('enc2',
                                    make_pipeline(ColumnExtractor(range(enc1_X_train.shape[1], 
                                                                        enc1_X_train.shape[1]+enc2_X_train.shape[1])),
                                                    clone(model)))]
            cv_stacking_k = 5                                                    
            if is_classifier(model):
                stacking_model = StackingClassifier(estimators=stacking_estimators, cv=cv_stacking_k)
                stacking_model.fit(concat_X_train_onlylabeled, y_train_onlylabeled)
                stacking_model_y_proba = stacking_model.predict_proba(concat_X_test)[:, 1]
            elif is_regressor(model):
                stacking_model = StackingRegressor(estimators=stacking_estimators, cv=cv_stacking_k)
                stacking_model.fit(concat_X_train_onlylabeled, y_train_onlylabeled)
                stacking_model_y_proba = stacking_model.predict(concat_X_test)
            else:
                raise ValueError('Model must be a classifier or regressor')
            
            pred_dict_st_lr.append({"y_proba": stacking_model_y_proba, "y_test": y_test,  "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})
        
        # Model using the mean of enc1 and enc2
        if not os.path.exists(st_mean_results_file):
            if not run_enc1:
                enc1_model = clone(model)
                enc1_model.fit(enc1_X_train_onlylabeled, y_train_onlylabeled)
                if is_classifier(model):
                    enc1_model_y_proba = enc1_model.predict_proba(enc1_X_test)[:, 1]
                elif is_regressor(model):
                    enc1_model_y_proba = enc1_model.predict(enc1_X_test)
                

            if not run_enc2:
                enc2_model = clone(model)
                enc2_model.fit(enc2_X_train_onlylabeled, y_train_onlylabeled)
                if is_classifier(model):
                    enc2_model_y_proba = enc2_model.predict_proba(enc2_X_test)[:, 1]
                elif is_regressor(model):
                    enc2_model_y_proba = enc2_model.predict(enc2_X_test)

            mean_proba = (enc1_model_y_proba + enc2_model_y_proba)/2
            pred_dict_st_mean.append({"y_proba": mean_proba, "y_test": y_test, "original_y_test": original_y_test, "train_len": len(y_train_onlylabeled)})
            
        # Print formatted taken time in hours, minutes and seconds
        print(f"\tExperiment with {enc1} and {enc2} using {labeled_percentage*100}% labeled instances (k={k}) took {time.strftime('%Hh %Mm %Ss', time.gmtime(time.time() - start))}")

    # Save dicts to pickle files
    if is_classifier(model):
        if not os.path.exists(ct_results_file):
            with open(ct_results_file, 'wb') as handle:
                pkl.dump(pred_dict_ct, handle, protocol=pkl.HIGHEST_PROTOCOL)

    if run_enc1:
        with open(enc1_results_file, 'wb') as handle:
            pkl.dump(pred_dict_enc1, handle, protocol=pkl.HIGHEST_PROTOCOL)

    if run_enc2:
        with open(enc2_results_file, 'wb') as handle:
            pkl.dump(pred_dict_enc2, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    if not os.path.exists(concat_results_file):
        with open(concat_results_file, 'wb') as handle:
            pkl.dump(pred_dict_concat, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    if not os.path.exists(st_lr_results_file):
        with open(st_lr_results_file, 'wb') as handle:
            pkl.dump(pred_dict_st_lr, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    if not os.path.exists(st_mean_results_file):
        with open(st_mean_results_file, 'wb') as handle:
            pkl.dump(pred_dict_st_mean, handle, protocol=pkl.HIGHEST_PROTOCOL)

    # Free memory (it seems threads are waiting taking memory innecesarily)
    del enc1_X
    del enc2_X
    del enc1_X_train
    del enc2_X_train
    del enc1_X_test
    del enc2_X_test
    del enc1_X_train_onlylabeled
    del enc2_X_train_onlylabeled
    del concat_X_train_onlylabeled
    del concat_X_test
    del y_train
    del y_train_ct
    del y_train_onlylabeled
    del y_test
    gc.collect()


if __name__ == "__main__":
    
    dataset_folder = sys.argv[1]
    dataset = dataset_folder.split('data/')[-1].split('/')[0]
    
    # model = DecisionTreeClassifier()
    # model = RandomForestClassifier()
    # model = LogisticRegression(max_iter=10000, n_jobs=1)
    # model = SVC(kernel='linear', probability=True)
    # model = MLPClassifier()
    model = LinearRegression(n_jobs=1)
    results_folder = f"results/multiview_experiments_{dataset}_{model.__class__.__name__}/"

    labeled_percentages = [0.5, 0.25, 0.15, 0.1, 0.05, 0.03, 0.01]
    #labeled_percentages = [0.15, 0.1, 0.05]
    
    y_file = os.path.join(dataset_folder, dataset+"_y.pkl")

    y = pkl.load(open(y_file, 'rb'))#[:1000]

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
        
        encodings_dict[encoding_name] = enc_X#[:1000]
    
    print(f"* Total dict size: {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024), 2)} MB | {round(sum([enc_X.nbytes for enc_X in encodings_dict.values()])/(1024*1024*1024), 2)} GB", flush=True)
    arguments = []
    for labeled_percentage in labeled_percentages:
        arguments.extend([(enc1, enc2, encodings_dict[enc1], encodings_dict[enc2], y, labeled_percentage, model, results_folder) for enc1, enc2 in combinations(encoding_names, 2)])
    print(f"* Total number of experiments: {len(arguments)}")
    print(f"* Number of cores: {sys.argv[2]}")
    print(f"* Starting experiments...")
    n_cores = int(sys.argv[2])
    with Pool(n_cores) as pool:
        pool.starmap(main, arguments, chunksize=1)
