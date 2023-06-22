import pickle as pk
import numpy as np
import pandas as pd
import random
import os
os.environ['OPENBLAS_NUM_THREADS'] = '20'
import sys
sys.path.append('../../../../sslearn')
from Bio import SeqIO
from sklearnex import patch_sklearn
# The names match scikit-learn estimators
patch_sklearn("SVC")
from sklearn.model_selection import StratifiedKFold, KFold
from models.MultiViewCoRegression import MultiviewCoReg

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Ridge

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, weightedtau
from scipy.stats import rankdata

from sklearn.model_selection import GridSearchCV
from scripts.MultiViewGridSearchCV import MultiViewGridSearchCV
from scripts.SSKFold import SSKFold, SSStratifiedKFold
from models.TriTrainingRegressor import TriTrainingRegressor
from datetime import datetime
from multiprocessing import Pool
import psutil
import warnings

import argparse

from sklearn.model_selection import (StratifiedKFold, StratifiedShuffleSplit,
                                     train_test_split)

def crossVal(dataset_folder, dataset_name, labeled_sizes, models, tune, log_file, train_variants, test_variants, random_state=1234): 

    args = []
    for model in models: 

        random.seed(random_state)
        
        for labeled_size in labeled_sizes:
            for i in range (1):
                args.append((i, dataset_folder, dataset_name, model, tune, labeled_size, log_file, train_variants, test_variants))
            
    with Pool(50) as pool: 
        pool.starmap(job, args, chunksize=1)

def job(i, dataset_folder, dataset_name, general_model, tune, labeled_size, log_file, train_variants, test_variants):
    
    # Read data JBA
    wt_sequence_file = os.path.join(dataset_folder, dataset_name+"_wt.fasta")
    wt_sequence = SeqIO.read(wt_sequence_file, "fasta").seq
    variants_dict = dict()
    X_seq_file = os.path.join(dataset_folder, dataset_name+"_X.pkl")
    X_file = os.path.join(dataset_folder, "X_One_hot.pkl")
    X = pk.load(open(X_file, 'rb'))
    y_file = os.path.join(dataset_folder, dataset+"_y.pkl")
    X_seq = pk.load(open(X_seq_file, 'rb'))
    y = pk.load(open(y_file, 'rb'))
    for i, seq in enumerate(X_seq):
        variants = sum([1 for i in range(len(seq)) if seq[i] != wt_sequence[i]])
        if variants in variants_dict:
            variants_dict[variants].append(i)
        else:
            variants_dict[variants] = [i]

    X = X.reshape(X.shape[0], -1)

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

    scores_dict = dict()
    predictions_dict = dict()
    tuned_params = dict()

    #split data 
    X_train, X_test = X[train_indexes], X[test_indexes]
    y_train, y_test = y[train_indexes], y[test_indexes]

    predictions_dict['y_test'] = y_test
    
    w = rankdata([-y for y in y_test])
    w = (w-np.min(w))/(np.max(w)-np.min(w))
    
    #Unlabeled instances
    labeled_indexes = np.random.choice(len(X_train), int(labeled_size * len(X_train)), replace=False)
    unlabeled_indexes = np.setdiff1d(np.arange(len(X_train)), labeled_indexes)
    

    if 'TriTrainingRegressor' == general_model: 
        y_train_tritr = np.copy(y_train)
        y_train_tritr[unlabeled_indexes] = np.nan

        #fit
        models = { 
            'rf': (RandomForestRegressor(), {'base_estimator__min_samples_split': range(3, 11), 
                                             'base_estimator__max_features': ['sqrt', 'log2', None],
                                             'error_tol': [0.0001, 0.01, 0.1, 1, 10], 
                                             'y_tol_per': [0.0001, 0.01, 0.1, 1, 10]}), 
            'ab': (AdaBoostRegressor(), {'base_estimator__n_estimators': range(50, 201, 25), 
                                         'base_estimator__learning_rate': 10. ** np.linspace(-4, 1, 30), 
                                         'error_tol': [0.0001, 0.01, 0.1, 1, 10], 
                                         'y_tol_per': [0.0001, 0.01, 0.1, 1, 10]}),
            'dt': (DecisionTreeRegressor(), {'base_estimator__min_samples_split': range(3, 11), 
                                             'base_estimator__max_features': ['sqrt', 'log2', None], 
                                             'error_tol': [0.0001, 0.01, 0.1, 1, 10], 
                                             'y_tol_per': [0.0001, 0.01, 0.1, 1, 10]}), 
            'r': (Ridge(), {'base_estimator__alpha': np.arange(0, 1.05, 0.05), 
                             'error_tol': [0.0001, 0.01, 0.1, 1, 10], 
                             'y_tol_per': [0.0001, 0.01, 0.1, 1, 10]}),
            'svm': (SVR(), {'base_estimator__kernel': ['rbf', 'linear'], 
                                             'error_tol': [0.0001, 0.01, 0.1, 1, 10], 
                                             'y_tol_per': [0.0001, 0.01, 0.1, 1, 10]}), 
        }
        
        for key in models: 
            print(datetime.now(), '--> TriTrainingRegressor + '+key+' (split ', i,'dataset', dataset_name,')')
            model = models[key][0]
            grid = models[key][1]
        
            tritr = TriTrainingRegressor(base_estimator=model)
            
            if tune:
                print(X_train.shape, y_train_tritr.shape)
                cv = SSKFold(n_splits=5, shuffle=True, u_symbol=np.nan)
                search = GridSearchCV(tritr, grid, cv=cv)
                result = search.fit(X_train, y_train_tritr)
                best_model = result.best_estimator_
                tuned_params['tritr_'+key] = result
                prediction_tritr = best_model.predict(X_test)

                print("--------------------------------------------------------------")
                print("Best parameters set found on development set for TriTrainingRegressor + "+key+":")
                print("Labeled size: ", labeled_size)
                print()
                print(result.best_params_)
                print()
                print("Best score: ", result.best_score_)
                print("Best Spearman: ", spearmanr(y_test, prediction_tritr)[0])
                print("--------------------------------------------------------------")
                with open(log_file, 'a') as best_params_file:
                    best_params_file.write("--------------------------------------------------------------\n")
                    best_params_file.write("Best parameters set found on development set for TriTrainingRegressor + "+str(key)+":\n")
                    best_params_file.write("Labeled size: " + str(labeled_size) + "\n")
                    best_params_file.write("\n")
                    best_params_file.write(str(result.best_params_) + "\n")
                    best_params_file.write("\n")
                    best_params_file.write("Best score: " + str(result.best_score_) + "\n")
                    best_params_file.write("Best Spearman: " + str(spearmanr(y_test, prediction_tritr)[0]) + "\n")
                    best_params_file.write("--------------------------------------------------------------\n")

            else: 
                tritr.fit(X_train, y_train_tritr)
                prediction_tritr = tritr.predict(X_test)

            #scores
            predictions_dict['prediction_tritr_'+key] = prediction_tritr
            scores_dict['mae_tritr_'+key] = mean_absolute_error(y_test, prediction_tritr)
            scores_dict['mse_tritr_'+key] = mean_squared_error(y_test, prediction_tritr)
            scores_dict['r2_tritr_'+key] = r2_score(y_test, prediction_tritr)
            scores_dict['spearman_tritr_'+key] = spearmanr(y_test, prediction_tritr)[0]
            scores_dict['wtau_tritr_'+key] = weightedtau(y_test, prediction_tritr)[0]
        
    
    if 'CoRegression' == general_model: 
        y_train_coreg = np.copy(y_train)
        y_train_coreg[unlabeled_indexes] = np.nan
        #fit
        print(datetime.now(), '--> CoRegression (split ', i,'dataset', dataset_name,')')

        cor = MultiviewCoReg(max_iters=100, pool_size=100)
        grid = [{'p1': [2], 'p2': [3, 4, 5]},
                {'p1': [3], 'p2': [4, 5]}, 
                {'p1': [4], 'p2': [5]}]
        cv = SSKFold(n_splits=5, shuffle=True, u_symbol=np.nan)
        search = GridSearchCV(cor, grid, cv=cv)
        result = search.fit(X_train, y_train_coreg)
        best_model = result.best_estimator_
        tuned_params['cor'] = result


        #scores
        prediction_cor = best_model.predict(X_test)
        predictions_dict['prediction_cor'] = prediction_cor

        print("--------------------------------------------------------------")
        print("Best parameters set found on development set for CoRegression:")
        print("Labeled size: ", labeled_size)
        print()
        print(result.best_params_)
        print()
        print("Best score: ", result.best_score_)
        print("Best Spearman: ", spearmanr(y_test, prediction_cor)[0])
        print("--------------------------------------------------------------")
        with open(log_file, 'a') as best_params_file: 
            best_params_file.write("--------------------------------------------------------------\n")
            best_params_file.write("Best parameters set found on development set for CoRegression:\n")
            best_params_file.write("Labeled size: " + str(labeled_size) + "\n")
            best_params_file.write("\n")
            best_params_file.write(str(result.best_params_) + "\n")
            best_params_file.write("\n")
            best_params_file.write("Best score: " + str(result.best_score_) + "\n")
            best_params_file.write("Best Spearman: " + str(spearmanr(y_test, prediction_cor)[0]) + "\n")
            best_params_file.write("--------------------------------------------------------------\n")

        scores_dict['mae_cor'] = mean_absolute_error(y_test, prediction_cor)
        scores_dict['mse_cor'] = mean_squared_error(y_test, prediction_cor)
        scores_dict['r2_cor'] = r2_score(y_test, prediction_cor)
        scores_dict['spearman_cor'] = spearmanr(y_test, prediction_cor)[0]
        scores_dict['wtau_cor'] = weightedtau(y_test, prediction_cor)[0]
    

if __name__=="__main__": 

    CLI=argparse.ArgumentParser()
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
        "--logfile",
        type=str
    )
    
    datasets_folder = "../../data/"
    dataset = 'avgfp'
    dataset_folder = datasets_folder+dataset+'/'

    log_file = CLI.parse_args().logfile

    models = ['TriTrainingRegressor', 'CoRegression']

    labeled_sizes = [0.5, 0.25, 0.1]

    tune = True

    crossVal(dataset_folder, dataset, labeled_sizes, models, tune=tune, log_file=log_file, train_variants=CLI.parse_args().trainvariants, test_variants=CLI.parse_args().testvariants, random_state=1234)