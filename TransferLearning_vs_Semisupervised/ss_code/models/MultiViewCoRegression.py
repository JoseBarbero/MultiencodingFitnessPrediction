from LAMDA_SSL.Base.InductiveEstimator import InductiveEstimator
from sklearn.base import RegressorMixin
import copy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import shuffle
import numpy as np

class MultiviewCoReg(InductiveEstimator,RegressorMixin):
    def __init__(self, k1=3, k2=3, p1=2, p2=5,
                 max_iters=100, pool_size=100,verbose=False, file=None):
        # >> Parameter
        # >> - k1: The k value for the k-nearest neighbors in the first base learner.
        # >> - k2: The k value for the k-nearest neighbors in the second base learner.
        # >> - p1: The order of the distance calculated in the first base learner.
        # >> - p2: The order of the distance calculated in the second base learner.
        # >> - max_iters: The maximum number of iterations.
        # >> - pool_size: The size of the buffer pool.
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.max_iters=max_iters
        self.pool_size=pool_size
        self.h1 = KNeighborsRegressor(n_neighbors=k1, p=p1, n_jobs=1)
        self.h2 = KNeighborsRegressor(n_neighbors=k2, p=p2, n_jobs=1)
        self.h1_temp = copy.copy(self.h1)
        self.h2_temp = copy.copy(self.h2)
        self.verbose = verbose
        self.file = file
        self.y_pred=None
        self._estimator_type = RegressorMixin._estimator_type


    def fit(self, X, y, X2=None):
        
        unlabeled_indexes = np.argwhere(np.isnan(y))
        labeled_indexes = [index for index in range(len(y)) if index not in unlabeled_indexes]
        
        X1 = copy.copy(X[labeled_indexes])
        unlabeled_X1 = copy.copy(X[unlabeled_indexes])
        
        if X2 is None: 
            X2=copy.copy(X[labeled_indexes])
            unlabeled_X2 = copy.copy(X[unlabeled_indexes])

        else: 
            # JBA se estaba pisando el X2 original, así que el unlabeled_X2 no tenía sentido
            X2_original = copy.copy(X2)
            X2=copy.copy(X2[labeled_indexes])
            unlabeled_X2 = copy.copy(X2_original[unlabeled_indexes])
        
        y1=copy.copy(y[labeled_indexes])
        y2=copy.copy(y[labeled_indexes])
        
        #fit models
        self.h1.fit(X1, y1)
        self.h2.fit(X2, y2)

        #select subset U' from U
        U_X1_pool, U_idx1_pool, U_X2_pool, U_idx2_pool = shuffle(unlabeled_X1, range(unlabeled_X1.shape[0]),
                                                                 unlabeled_X2, range(unlabeled_X2.shape[0]))
        U_X1_pool, U_X2_pool = U_X1_pool[:self.pool_size], U_X2_pool[:self.pool_size]
        U_idx1_pool, U_idx2_pool = U_idx1_pool[:self.pool_size], U_idx2_pool[:self.pool_size]
        
        for i in range(self.max_iters):
            stop_training = True
            to_remove=[]
            for idx_h in [1, 2]:
                if idx_h == 1:
                    h = self.h1
                    h_temp = self.h1_temp
                    L_X, L_y = X1, y1
                    U_X_pool = U_X1_pool
                else:
                    h = self.h2
                    h_temp = self.h2_temp
                    L_X, L_y = X2, y2
                    U_X_pool = U_X2_pool
                
                deltas = np.zeros((U_X_pool.shape[0],))
                y_u_hats = np.zeros((U_X_pool.shape[0],))
                for idx_u, x_u in enumerate(U_X_pool): 
                    x_u = x_u.reshape(1,-1)
                    y_u_hat = h.predict(x_u)
                    omega = h.kneighbors(x_u, return_distance=False)[0]
                    X_temp = np.concatenate((L_X, x_u))
                    y_temp = np.concatenate((L_y, y_u_hat))
                    h_temp.fit(X_temp, y_temp)
                    
                    delta = 0
                    for idx_o in omega:
                        delta += (L_y[idx_o] -
                                  h.predict(L_X[idx_o].reshape(1, -1))) ** 2
                        
                        delta -= (L_y[idx_o] -
                                  h_temp.predict(L_X[idx_o].reshape(1, -1))) ** 2
                        
                    deltas[idx_u] = delta
                    y_u_hats[idx_u] = y_u_hat
                
                #choose bigger delta 
                max_idx = np.argmax(deltas)
                
                #if bigger delta is greater than 0
                if deltas[max_idx] > 0:
                    stop_training=False

                    if idx_h == 1:
                        x_u = U_X2_pool[max_idx].reshape(1, -1)
                        y_u_hat = y_u_hats[max_idx]
                        idx_u=U_idx2_pool[max_idx]
                        X2 = np.concatenate((X2, x_u))
                        y2 = np.append(y2, y_u_hat)
                    else:
                        x_u = U_X1_pool[max_idx].reshape(1, -1)
                        y_u_hat = y_u_hats[max_idx]
                        idx_u=U_idx1_pool[max_idx]
                        X1 = np.concatenate((X1, x_u))
                        y1 = np.append(y1, y_u_hat)
                    to_remove.append(idx_u)

            if stop_training:
                break
            else:
                #refit 
                self.h1.fit(X1, y1)
                self.h2.fit(X2, y2)

                #delete labeled instance from U
                unlabeled_X1 = np.delete(unlabeled_X1, to_remove, axis=0)
                unlabeled_X2 = np.delete(unlabeled_X2, to_remove, axis=0)

                #select subset U' from U
                U_X1_pool, U_idx1_pool, U_X2_pool, U_idx2_pool = shuffle(unlabeled_X1, range(unlabeled_X1.shape[0]), 
                                                                         unlabeled_X2, range(unlabeled_X2.shape[0]))
                U_X1_pool, U_X2_pool = U_X1_pool[:self.pool_size], U_X2_pool[:self.pool_size]
                U_idx1_pool, U_idx2_pool = U_idx1_pool[:self.pool_size], U_idx2_pool[:self.pool_size]
                
        return self

    
    def predict(self, X, X2=None):
        if X2 is not None: 
            result1 = self.h1.predict(X)
            result2 = self.h2.predict(X2)
        else: 
            result1 = self.h1.predict(X)
            result2 = self.h2.predict(X)
        result = 0.5 * (result1 + result2)
        return result