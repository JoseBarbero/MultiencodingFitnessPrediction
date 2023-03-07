# Include sslearn folder
import sys
sys.path.append('../../../sslearn')
from sslearn.wrapper import CoTraining
from sslearn.base import OneVsRestSSLClassifier
import pickle as pkl
import numpy as np
import os
from SequenceEncoding import SequenceEncoding
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import combinations
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit


labeled_percentage = 0.5
enc1_X = pkl.load(open("../data/BRCA1_HUMAN_Fields2015_y2h/X_One_hot_6_bit.pkl", 'rb'))
enc2_X = pkl.load(open("../data/BRCA1_HUMAN_Fields2015_y2h/X_Binary_5_bit.pkl", 'rb'))
y = pkl.load(open("../data/BRCA1_HUMAN_Fields2015_y2h/BRCA1_HUMAN_Fields2015_y2h_y.pkl", 'rb'))
y = np.where(y >= np.percentile(y, 75), 1, 0).ravel()
# Stratified kfold
skf = StratifiedKFold(n_splits=5, shuffle=True)

results_df = pd.DataFrame(columns=['View', 'max_iterations', 'poolsize', 'positives', 'negatives', 'experimental', 'force_second_view', 'AUC'])

for train_index, test_index in skf.split(enc1_X, y):
    enc1_X_train, enc1_X_test = enc1_X[train_index], enc1_X[test_index]
    enc2_X_train, enc2_X_test = enc2_X[train_index], enc2_X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Split train set into labeled and unlabeled
    sss = StratifiedShuffleSplit(n_splits=1, test_size=labeled_percentage)
    unlabeled_indexes, labeled_indexes = next(sss.split(enc1_X_train, y_train))

    # Get unlabeled subsets
    enc1_X_train_onlylabeled = np.delete(enc1_X_train, unlabeled_indexes, axis=0)
    enc2_X_train_onlylabeled = np.delete(enc2_X_train, unlabeled_indexes, axis=0)
    y_train_onlylabeled = np.delete(y_train, unlabeled_indexes, axis=0)

    # Set unlabeled_indexes to -1 in y
    y_train_ct = np.copy(y_train)
    y_train_ct[unlabeled_indexes] = -1
    
    enc1_model = RandomForestClassifier()
    enc1_model.fit(enc1_X_train_onlylabeled, y_train_onlylabeled)
    enc1_model_y_proba = enc1_model.predict_proba(enc1_X_test)[:, 1]
    enc1_model_auc = roc_auc_score(y_test, enc1_model_y_proba)
    results_df = pd.concat([results_df, pd.DataFrame({'View': 'One_hot', 'max_iterations': 0, 'poolsize': 0, 'positives': 0, 'negatives': 0, 'experimental': False, 'force_second_view': False, 'AUC': enc1_model_auc}, index=[0])])
    

    enc2_model = RandomForestClassifier()
    enc2_model.fit(enc2_X_train_onlylabeled, y_train_onlylabeled)
    enc2_model_y_proba = enc2_model.predict_proba(enc2_X_test)[:, 1]
    enc2_model_auc = roc_auc_score(y_test, enc2_model_y_proba)
    results_df = pd.concat([results_df, pd.DataFrame({'View': 'ANN4D', 'max_iterations': 0, 'poolsize': 0, 'positives': 0, 'negatives': 0, 'experimental': False, 'force_second_view': False, 'AUC': enc2_model_auc}, index=[0])])


    # Grid search for CoTraining using cotraining_parameters
    # Get each combination of values from cotrainin_parameters dict
    max_iterations=30
    poolsize=10
    positives=2
    negatives=3
    experimental=False

    ct = CoTraining(base_estimator=RandomForestClassifier(),
                    max_iterations=max_iterations,
                    poolsize=poolsize,
                    positives=positives,
                    negatives=negatives,
                    experimental=experimental,
                    force_second_view=True
                    )
    ct.fit(enc1_X_train, y_train_ct, enc1_X_test, enc2_X_test, y_test, X2=enc2_X_train)
    ct_y_proba = ct.predict_proba(enc1_X_test, X2=enc2_X_test)[:, 1]
    ct_auc = roc_auc_score(y_test, ct_y_proba)
    results_df = pd.concat([results_df, pd.DataFrame({'View': 'CoTraining', 'max_iterations': max_iterations, 'poolsize': poolsize, 'positives': positives, 'negatives': negatives, 'experimental': experimental, 'force_second_view': True, 'AUC': ct_auc}, index=[0])])

    print(f"View1 AUC: {enc1_model_auc}")
    print(f"View2 AUC: {enc1_model_auc}")
    print(f"CoTraining AUC: {ct_auc}. [max_iteration={max_iterations}, poolsize={poolsize}, positives={positives}, negatives={negatives}, experimental={experimental}]")
    break
# Save df to file
#results_df.to_csv("../CoTrainingGridSearch_results_df.csv", index=False)