from ast import arguments
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from SequenceEncoding import SequenceEncoding
import pickle as pkl
import sys
import os
import numpy as np
import warnings
import time
from sklearn.base import TransformerMixin
from itertools import combinations
from multiprocessing import Pool
import pandas as pd

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

def gather_results(encoding_pairs, final_classifiers, results_folder):
    df = pd.DataFrame(columns=['Enc1', 'Enc2', 'Classifier', 'Stacking', 'AUC'])
    for enc1, enc2 in encoding_pairs:
        for final_classifier_name in final_classifiers:
            results_dict = pkl.load(open(os.path.join(results_folder, f"{final_classifier_name}_{enc1}_{enc2}_stacking_results.pkl"), "rb"))
            for clf, st_types in results_dict.items():
                for st_type, aucs in st_types.items():
                    for auc in aucs:
                        df = pd.concat([df, pd.DataFrame({'Enc1': enc1, 'Enc2': enc2, 'Classifier': clf, 'Stacking': st_type, 'AUC': auc}, index=[0])], ignore_index=True)
    return df
    

def main(enc1, enc2, enc1_X, enc2_X, y, final_classifier_name, final_classifier, results_folder):
        
    results_dict = dict()
        
    results_dict[final_classifier_name] = dict()
    results_dict[final_classifier_name]["clf"] = []
    results_dict[final_classifier_name]["st"] = []
    results_dict[final_classifier_name]["st+"] = []
    results_dict[final_classifier_name]["mean"] = []

    # Create results folder if it doesn't exist
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in skf.split(enc1_X, y):
        print("Final Classifier: ", final_classifier_name, "Fold: ", len(results_dict[final_classifier_name]["clf"])+1)
        enc1_X_train, enc1_X_test = enc1_X[train_index], enc1_X[test_index]
        enc2_X_train, enc2_X_test = enc2_X[train_index], enc2_X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        selected_cls = RandomForestClassifier()

        st_estimators = [('enc1', make_pipeline(ColumnExtractor(range(0,enc1_X_train.shape[1])),
                                                    selected_cls)),
                        ('enc2', make_pipeline(ColumnExtractor(range(enc1_X_train.shape[1],
                                                                        enc1_X_train.shape[1]+enc2_X_train.shape[1])),
                                                    selected_cls))
                                ]
        st_plus_estimators = [('enc1', make_pipeline(ColumnExtractor(range(0,enc1_X_train.shape[1])),
                                                            selected_cls)),
                                ('enc2', make_pipeline(ColumnExtractor(range(enc1_X_train.shape[1],
                                                                                enc1_X_train.shape[1]+enc2_X_train.shape[1])),
                                                            selected_cls)),
                                ('all',selected_cls)
                                ]

        clf_stack = StackingClassifier(estimators=st_estimators, 
                                final_estimator=final_classifier)

        clf_stack_plus = StackingClassifier(estimators=st_plus_estimators, 
                                final_estimator=final_classifier)

        concat_X_train = np.concatenate((enc1_X_train, enc2_X_train), axis=1)
        concat_X_test = np.concatenate((enc1_X_test, enc2_X_test), axis=1)
        
        selected_cls.fit(concat_X_train, y_train)
        cls_y_pred = selected_cls.predict_proba(concat_X_test)[:,1]
        results_dict[final_classifier_name]["clf"].append(roc_auc_score(y_test, cls_y_pred))

        clf_stack.fit(concat_X_train, y_train)
        stack_y_pred = clf_stack.predict_proba(concat_X_test)[:,1]
        results_dict[final_classifier_name]["st"].append(roc_auc_score(y_test, stack_y_pred))

        clf_stack_plus.fit(concat_X_train, y_train)
        stack_plus_y_pred = clf_stack_plus.predict_proba(concat_X_test)[:,1]
        results_dict[final_classifier_name]["st+"].append(roc_auc_score(y_test, stack_plus_y_pred))

        #Model view 1
        view1_cls = selected_cls.fit(enc1_X_train, y_train)
        view2_cls = selected_cls.fit(enc2_X_train, y_train)
        # Get the mean from view1_cls and view2_cls prodict_proba 
        view1_proba = view1_cls.predict_proba(enc1_X_test)[:, 1]
        view2_proba = view2_cls.predict_proba(enc2_X_test)[:, 1]
        mean_proba = (view1_proba + view2_proba)/2
        results_dict[final_classifier_name]["mean"].append(roc_auc_score(y_test, mean_proba))

    # Save results to file
    with open(os.path.join(results_folder, f"{final_classifier_name}_{enc1}_{enc2}_stacking_results.pkl"), 'wb') as handle:
        pkl.dump(results_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    
    dataset = "avgfp"
    dataset_folder = os.path.join("../data/", dataset)
    
    results_folder = f"stacking_selection_experiments_{dataset}/"
    
    y_file = os.path.join(dataset_folder, dataset+"_y.pkl")
    y = pkl.load(open(y_file, 'rb'))
    # Change regression labels to binary labels above first quartile and below
    y = np.where(y >= np.percentile(y, 75), 1, 0).ravel()

    encoding_names = ["One_hot", "One_hot_6_bit", "Binary_5_bit", "Hydrophobicity_matrix",
             "Meiler_parameters", "Acthely_factors", "PAM250", "BLOSUM62",
             "Miyazawa_energies", "Micheletti_potentials", "AESNN3",
             "ANN4D", "ProtVec"]

    #encoding_pairs = list(combinations(encoding_names, 2))
    encoding_pairs = [("PAM250", "Micheletti_potentials")]

    final_classifiers = [("LogisticRegression5000", LogisticRegression(max_iter=5000)),
                          ("LogisticRegression25000", LogisticRegression(max_iter=25000)),
                          ("RandomForestClassifier100", RandomForestClassifier(n_estimators=100)),
                          ("RandomForestClassifier1000", RandomForestClassifier(n_estimators=1000))]
    
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
    arguments_list = []
    for final_classifier in final_classifiers:
        arguments_list.extend([(enc1, enc2, encodings_dict[enc1], encodings_dict[enc2], y, final_classifier[0], final_classifier[1], results_folder) for enc1, enc2 in encoding_pairs])
    
    # Ignore warnings
    n_cores = int(sys.argv[1])
    with Pool(n_cores) as pool:
        pool.starmap(main, arguments_list)

    results_df = gather_results(encoding_pairs, [name for name, clf in final_classifiers], results_folder)
    # Serialize in pickle
    with open(os.path.join(results_folder, f"stacking_results.pkl"), 'wb') as handle:
        pkl.dump(results_df, handle, protocol=pkl.HIGHEST_PROTOCOL)