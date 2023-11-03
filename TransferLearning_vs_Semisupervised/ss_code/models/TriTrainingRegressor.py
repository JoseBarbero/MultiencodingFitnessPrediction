from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator
import sys
sys.path.insert(1, '/home/aolivares/sslearn')

from sslearn.wrapper import TriTraining

from sklearn.utils import check_random_state, check_X_y, resample
from sslearn.utils import check_n_jobs, safe_division

from sklearn.base import clone as skclone
import numpy as np
from joblib import Parallel, delayed
import math



def get_regression_dataset(X, y):
    #X, y = check_X_y(X, y)
    
    X_label = [_x for _x, _y in zip(X, y) if not np.isnan(_y)]
    y_label = [_y for _y in y  if not np.isnan(_y)]
    X_unlabel = [_x for _x, _y in zip(X, y) if np.isnan(_y)]
       
    return np.array(X_label), np.array(y_label), np.array(X_unlabel)


class TriTrainingRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        base_estimator=DecisionTreeRegressor(),
        n_samples=None, 
        y_tol_per = 1, 
        random_state=None,
        n_jobs=None,
    ):
        """TriTraining
        Zhi-Hua Zhou and Ming Li,
        "Tri-training: exploiting unlabeled data using three classifiers,"
        in <i>IEEE Transactions on Knowledge and Data Engineering</i>,
        vol. 17, no. 11, pp. 1529-1541, Nov. 2005,
        doi: 10.1109/TKDE.2005.186.
        Parameters
        ----------
        base_estimator : RegressorMixin, optional
            An estimator object implementing fit and predict, by default DecisionTreeRegressor()
        n_samples : int, optional
            Number of samples to generate.
            If left to None this is automatically set to the first dimension of the arrays., by default None
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        """
        self.base_estimator = base_estimator
        self.n_samples = n_samples
        self.y_tol_per = y_tol_per
        self.y_tol = None
        self._N_LEARNER = 3
        self._epsilon = sys.float_info.epsilon
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, **kwards):
        """Build a TriTraining regressor from the training set (X, y).
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values, None if unlabeled.
        Returns
        -------
        self: TriTraining
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)
        self.n_jobs = check_n_jobs(self.n_jobs)

        X_label, y_label, X_unlabel = get_regression_dataset(X, y)

        hypotheses = []
        e_ = [0.5] * self._N_LEARNER
        l_ = [0] * self._N_LEARNER

        for _ in range(self._N_LEARNER):
            X_sampled, y_sampled = resample(
                X_label,
                y_label,
                replace=True,
                n_samples=self.n_samples,
                random_state=random_state,
            )

            hypotheses.append(
                skclone(self.base_estimator).fit(X_sampled, y_sampled, **kwards)
            )

        something_has_changed = True
        
        #CAMBIO: diferencia mínima entre valores de y para considerarse iguales 
        self.y_tol = self.y_tol_per*(np.min(y_label)-np.max(y_label))

        while something_has_changed:
            something_has_changed = False
            L = [[]] * self._N_LEARNER
            Ly = [[]] * self._N_LEARNER
            e = []
            updates = [False] * 3

            for i in range(self._N_LEARNER):
                hj, hk = TriTraining._another_hs(hypotheses, i)
                e.append(
                    #CAMBIO: forma de calcular el error 
                    self._measure_error(X_label, y_label, hj, hk, self._epsilon)
                )
                if e_[i] <= e[i]:
                    continue
                y_p = hj.predict(X_unlabel)
                #CAMBIO: forma de saber si ambas predicciones son "iguales"
                validx = np.isclose(y_p, hk.predict(X_unlabel), atol = self.y_tol)
    
                L[i] = X_unlabel[validx]
                Ly[i] = y_p[validx]

                if l_[i] == 0:
                    l_[i] = math.floor(
                        safe_division(e[i], (e_[i] - e[i]), self._epsilon) + 1
                    )
                if l_[i] >= len(L[i]):
                    continue
                if e[i] * len(L[i]) < e_[i] * l_[i]:
                    updates[i] = True
                elif l_[i] > safe_division(e[i], e_[i] - e[i], self._epsilon):
                    L[i], Ly[i] = TriTraining._subsample(
                        (L[i], Ly[i]),
                        math.ceil(
                            safe_division(e_[i] * l_[i], e[i], self._epsilon) - 1
                        ),
                        random_state,
                    )
                    updates[i] = True

            hypotheses = Parallel(n_jobs=self.n_jobs)(
                delayed(self._fit_estimator)(
                    hypotheses[i], X_label, y_label, L[i], Ly[i], updates[i], **kwards
                )
                for i in range(self._N_LEARNER)
            )

            for i in range(self._N_LEARNER):
                if updates[i]:
                    e_[i] = e[i]
                    l_[i] = len(L[i])
                    something_has_changed = True

        self.h_ = hypotheses
        self.columns_ = [list(range(X.shape[1]))] * self._N_LEARNER

        return self

    def _fit_estimator(self, hyp, X_label, y_label, L, Ly, update, **kwards):
        if update:
            _tempL = np.concatenate((X_label, L))
            _tempY = np.concatenate((y_label, Ly))

            return hyp.fit(_tempL, _tempY, **kwards)
        return hyp

    @staticmethod
    def _another_hs(hs, index):
        """Get the other hypotheses
        Parameters
        ----------
        hs : list
            hypotheses collection
        index : int
            base hypothesis  index
        Returns
        -------
        list
        """
        another_hs = []
        for i in range(len(hs)):
            if i != index:
                another_hs.append(hs[i])
        return another_hs

    @staticmethod
    def _subsample(L, s, random_state=None):
        """Randomly removes |L| - s number of examples from L
        Parameters
        ----------
        L : tuple of array-like
            Collection pseudo-labeled candidates and its labels
        s : int
            Equation 10 in paper
        random_state : int, RandomState instance, optional
            controls the randomness of the estimator, by default None
        Returns
        -------
        tuple
            Collection of pseudo-labeled selected for enlarged labeled examples.
        """
        to_remove = len(L[0]) - s
        select = len(L[0]) - to_remove

        return resample(*L, replace=False, n_samples=select, random_state=random_state)
    
    @staticmethod
    def _are_same_label(y1, y2, y_tol): 
        return np.isclose(y1, y2, atol = y_tol)

    #CAMBIO: forma de calcular el error 
    def _measure_error(
        self, X, y, h1: RegressorMixin, h2: RegressorMixin, epsilon=sys.float_info.epsilon, **kwards
    ):
        """Calculate the error between two hypotheses
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training labeled input samples.
        y : array-like of shape (n_samples,)
            The target values.
        h1 : RegressorMixin
            First hypothesis
        h2 : RegressorMixin
            Second hypothesis
        epsilon : float
            A small number to avoid division by zero
        Returns
        -------
        float
            Division of the number of labeled examples on which both h1 and h2 make incorrect classification,
            by the number of labeled examples on which the classification made by h1 is the same as that made by h2.
        """
        y1 = h1.predict(X).astype('float64')
        y2 = h2.predict(X).astype('float64')
        
        predict_same = TriTrainingRegressor._are_same_label(y1, y2, self.y_tol)
        predict_wrong = np.logical_not(TriTrainingRegressor._are_same_label(y2, y, self.y_tol))
                                     
        error = np.count_nonzero(np.logical_and(predict_same, predict_wrong))
        coincidence = np.count_nonzero(predict_same)
   
        return safe_division(error, coincidence, epsilon)
    
    #CAMBIO: ???
    def predict(self, X): 
        predictions = []
        for h in self.h_: 
            predictions.append(h.predict(X))       
        return np.mean(predictions, axis=0)