from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import _num_samples

import numpy as np


class SSKFold(KFold): 
    
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None, u_symbol=None):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        self.u_symbol = u_symbol
        self.indices_map = dict()
        
    def split(self, X, y=None, groups=None): 
        
        if np.isnan(self.u_symbol):
            labeled = [i for i, _y in enumerate(y) if not np.isnan(_y)]
        else:
            labeled = [i for i, _y in enumerate(y) if self.u_symbol != _y]
        unlabeled = [i for i in range(len(y)) if i not in labeled]
        
        sorted_indices = np.concatenate((labeled, unlabeled), axis=0)
        sorted_X = np.concatenate((X[labeled], X[unlabeled]))
        unsorted_indices = np.arange(_num_samples(sorted_X))
        
        for s_i, u_i in zip(sorted_indices, unsorted_indices): 
            self.indices_map[u_i] = s_i
            
        for train_index, test_index in super().split(X[labeled], y[labeled], groups): 
            train_index = [self.indices_map[train_i] for train_i in train_index]
            train_index = train_index + unlabeled
            test_index = [self.indices_map[test_i] for test_i in test_index]
            
            yield train_index, test_index
            
class SSStratifiedKFold(StratifiedKFold): 
    
    def __init__(self, n_splits=5, *, shuffle=False, random_state=None, u_symbol=-1):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        self.u_symbol = u_symbol
        self.indices_map = dict()
            
    def split(self, X, y=None, groups=None): 
        
        labeled = [i for i, _y in enumerate(y) if _y != self.u_symbol]
        unlabeled = [i for i in range(len(y)) if i not in labeled]
        
        sorted_indices = np.concatenate((labeled, unlabeled), axis=0)
        sorted_X = np.concatenate((X[labeled], X[unlabeled]))
        unsorted_indices = np.arange(_num_samples(sorted_X))
        
        for s_i, u_i in zip(sorted_indices, unsorted_indices): 
            self.indices_map[u_i] = s_i
            
        for train_index, test_index in super().split(X[labeled], y[labeled], groups): 
            train_index = [self.indices_map[train_i] for train_i in train_index]
            train_index = train_index + unlabeled
            test_index = [self.indices_map[test_i] for test_i in test_index]
            
            yield train_index, test_index
            
if __name__=="__main__":  
    
    print('\n######### NO STRATIFIED #########\n')
    X = np.array([[1, 3, 7, 5, 8], 
                 [5, 6, 7, 9, 3],
                 [6, 7, 8, 9, 0], 
                 [2, 4, 6, 7, 8], 
                 [3, 8, 6, 5, 1], 
                 [5, 7, 2, 5, 9], 
                 [2, 7, 8, 0, 8], 
                 [5, 1, 5, 8, 9]])
    
    y = np.array([None, 44, None, 67, 78, 34, 27, 87])
    
    print('X:')
    print(X)
    print('y:')
    print(y)
    
    cv = SSKFold(n_splits = 2)
    for train, test in cv.split(X, y): 
        print('*******NEW SPLIT*******:')
        print('train X:')
        print(X[train])
        print('train y: ')
        print(y[train])
        print('test X:')
        print(X[test])
        print('test y: ')
        print(y[test])

    print('\n######### STRATIFIED #########\n')
    X = np.array([[1, 3, 7, 5, 8], 
                 [5, 6, 7, 9, 3],
                 [6, 7, 8, 9, 0], 
                 [2, 4, 6, 7, 8], 
                 [3, 8, 6, 5, 1], 
                 [5, 7, 2, 5, 9], 
                 [2, 7, 8, 0, 8], 
                 [5, 1, 5, 8, 9]])
    
    y = np.array([-1, 1, -1, 1, 2, 2, 1, 2])
    
    print('X:')
    print(X)
    print('y:')
    print(y)
    
    cv = SSStratifiedKFold(n_splits = 2)
    for train, test in cv.split(X, y): 
        print('*******NEW SPLIT*******:')
        print('train X:')
        print(X[train])
        print('train y: ')
        print(y[train])
        print('test X:')
        print(X[test])
        print('test y: ')
        print(y[test])