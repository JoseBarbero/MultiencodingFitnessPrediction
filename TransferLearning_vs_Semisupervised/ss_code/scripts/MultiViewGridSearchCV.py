from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.metrics import check_scoring
from sklearn.utils.validation import indexable, _check_fit_params, _num_samples
from sklearn.base import clone
from joblib import Parallel
from collections import defaultdict, Counter
from itertools import product
from sklearn.utils.fixes import delayed
import numbers
import time
import warnings
from traceback import format_exc
from contextlib import suppress

def _fit_and_score(
    estimator, 
    X, 
    y, 
    X2, 
    scorer, 
    train, 
    test, 
    verbose, 
    parameters, 
    fit_params, 
    return_train_score=False, 
    return_parameters=False, 
    return_n_test_samples=False,
    return_times=False, 
    return_estimator=False, 
    split_progress=None, 
    candidate_progress=None, 
    error_score=np.nan,
): 

    if not isinstance(error_score, numbers.Number) and error_score != "raise": 
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )
        
    progress_msg = ""
    if verbose > 2: 
        if split_progress is not None: 
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9: 
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"
            
    if verbose > 1: 
        if parameters is None: 
            params_msg = ""
        else: 
            sorted_keys = sorted(parameters)
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
            
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")
        
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)
    
    if parameters is not None: 
        cloned_parameters = {}
        for k, v in parameters.items(): 
            cloned_parameters[k] = clone(v, safe=False)
            
        estimator = estimator.set_params(**cloned_parameters)
        
    start_time = time.time()
    
    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    X2_train, X2_test = X2[train], X2[test]
    
    result = {}
    try: 
        if y_train is None: 
            estimator.fit(X_train, X2=X2_train, **fit_params)
        else: 
            estimator.fit(X_train, y_train, X2=X2_train, **fit_params)
            
    except Exception: 
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise": 
            raise
        elif isinstance(error_score, numbers.Number): 
            if isinstance(scorer, dict): 
                test_scores = {name: error_score for name in scorer}
                if return_train_score: 
                    train_scorers = test_scores.copy()
            else: 
                test_scores = error_score
                if return_train_score: 
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else: 
        result["fit_error"] = None
        
        fit_time = time.time() - start_time
        test_scores = _score(estimator, X_test, y_test, X2_test, scorer, error_score)
        score_time = time.time() - start_time - fit_time
        if return_train_score: 
            train_scores = _score(estimator, X_train, y_train, X2_test, scorer, error_score)
            
    if verbose > 1: 
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2: 
            if isinstance(test_scores, dict): 
                for scorer_name in sorted(test_scores): 
                    result_msg += f" {scorer_name}: ("
                    if return_train_score: 
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg -= f"test={test_scores[scorer_name]:.3f})"
            else: 
                result_msg += ", score="
                if return_train_score: 
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else: 
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"
        
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
        
    return result

def _score(estimator, X_test, y_test, X2_test, scorer, error_score="raise"): 
    try: 
        if y_test is None: 
            scores = scorer(estimator, X_test, X2_test)
        else: 
            scores = scorer(estimator, X_test, y_test, X2=X2_test)
    except Exception: 
        if error_score == "raise": 
            raise
        else: 
            scores = error_score
            warnings.warn(
                "Scoring failed. The score on this train-test partition for "
                f"these parameters will be set to {error_score}. Details: \n"
                f"{format_exc()}",
                UserWarning, 
            )
    error_msg = "scoring must return a number, got %s (%s) instead. (scorer=%s)"
    if isinstance(scores, dict): 
        for name, score in scores.items(): 
            if hasattr(score, "item"): 
                with suppress(ValueError): 
                    score = score.item()
            if not isinstance(score, numbers.Number): 
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else: 
        if hasattr(scores, "item"): 
            with suppress(ValueError): 
                scores = scores.item()
        if not isinstance(scores, numbers.Number): 
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores
        
    
    
def _warn_or_raise_about_fit_failures(results, error_score):
    fit_errors = [
        result["fit_error"] for result in results if result["fit_error"] is not None
    ]
    if fit_errors:
        num_failed_fits = len(fit_errors)
        num_fits = len(results)
        fit_errors_counter = Counter(fit_errors)
        delimiter = "-" * 80 + "\n"
        fit_errors_summary = "\n".join(
            f"{delimiter}{n} fits failed with the following error:\n{error}"
            for error, n in fit_errors_counter.items()
        )

        if num_failed_fits == num_fits:
            all_fits_failed_message = (
                f"\nAll the {num_fits} fits failed.\n"
                "It is very likely that your model is misconfigured.\n"
                "You can try to debug the error by setting error_score='raise'.\n\n"
                f"Below are more details about the failures:\n{fit_errors_summary}"
            )
            raise ValueError(all_fits_failed_message)

        else:
            some_fits_failed_message = (
                f"\n{num_failed_fits} fits failed out of a total of {num_fits}.\n"
                "The score on these train-test partitions for these parameters"
                f" will be set to {error_score}.\n"
                "If these failures are not expected, you can try to debug them "
                "by setting error_score='raise'.\n\n"
                f"Below are more details about the failures:\n{fit_errors_summary}"
            )
            warnings.warn(some_fits_failed_message, FitFailedWarning)

class MultiViewGridSearchCV(GridSearchCV): 
    
    def __init__(
        self, 
        estimator, 
        param_grid, 
        *, 
        scoring=None, 
        n_jobs=None, 
        refit=True, 
        cv=None, 
        verbose=0, 
        pre_dispatch="2*n_jobs", 
        error_score=np.nan, 
        return_train_score=False,
    ): 
        super().__init__(
            estimator=estimator, 
            param_grid=param_grid,
            scoring=scoring, 
            n_jobs=n_jobs, 
            refit=refit, 
            cv=cv, 
            verbose=verbose, 
            pre_dispatch=pre_dispatch, 
            error_score=error_score, 
            return_train_score=return_train_score, 
        )
        
    def fit(self, X, y=None, X2=None, *, groups=None, **fit_params): 
        
        estimator = self.estimator
        refit_metric = "score"
        
        if callable(self.scoring): 
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str): 
            scorers = check_scoring(self.estimator, self.scoring)
        else: 
            scorers = _check_multimetric_scoring(self.estimator, self.scoring)
            self._check_refit_for_multimetric(scorers)
            refit_metric = self.refit
            
        X, y, groups = indexable(X, y, groups)
        X2, y, groups = indexable(X2, y, groups)
        #fit_params = _check_fit_params(X, fit_params)

        n_splits = self.cv.n_splits 
        
        base_estimator = clone(self.estimator)
        
        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)
        
        fit_and_score_kwargs = dict(scorer=scorers, 
                                    fit_params=fit_params, 
                                    return_train_score=self.return_train_score,
                                    return_n_test_samples=True, 
                                    return_times=True, 
                                    return_parameters=False,
                                    error_score=self.error_score, 
                                    verbose=self.verbose,
                                   )
        results = {}
        with parallel: 
            all_candidate_params = []
            all_out = []
            all_more_results = defaultdict(list)
            
            def evaluate_candidates(candidate_params, cv=self.cv, more_results=None): 
                cv = cv
                candidate_params = list(candidate_params)
                n_candidates = len(candidate_params)
                
                if self.verbose > 0: 
                    print(
                        "Fitting {0} folds for each of {1} candidates,"
                        " totalling {2} fits".format(
                            n_splits, n_candidates, n_candidates * n_splits
                        )
                    )
                out = parallel(
                    delayed(_fit_and_score)(
                        clone(base_estimator), 
                        X, 
                        y, 
                        X2, 
                        train=train, 
                        test=test, 
                        parameters=parameters, 
                        split_progress=(split_idx, n_splits), 
                        candidate_progress=(cand_idx, n_candidates), 
                        **fit_and_score_kwargs,
                    )
                    for (cand_idx, parameters), (split_idx, (train, test)) in product(
                        enumerate(candidate_params), enumerate(cv.split(X, y, groups))
                    )
                )
                
                if len(out) < 1: 
                    raise ValueError(
                        "No fits were performed. "
                        "Was the CV iterator empty? "
                        "Were there no candidates?"
                    )
                elif len(out) != n_candidates * n_splits: 
                    raise ValueError(
                        "cv.split and cv.get_n_splits returned "
                        "inconsistent results. Expected {} "
                        "splits, got {}".format(n_splits, len(out) // n_candidates)
                    )
                
                _warn_or_raise_about_fit_failures(out, self.error_score)
                
                if callable(self.scoring): 
                    _instert_error_scores(out, self.error_score)
                    
                all_candidate_params.extend(candidate_params)
                all_out.extend(out)
                
                if more_results is not None: 
                    for key, value in more_results.items(): 
                        all_more_results[key].extend(value)
                        
                nonlocal results
                results = self._format_results(
                    all_candidate_params, n_splits, all_out, all_more_results
                )

                return results
            
            self._run_search(evaluate_candidates)
        
            first_test_score = all_out[0]["test_scores"]
            self.multmetric_ = isinstance(first_test_score, dict)

            if callable(self.scoring) and self.multimetric_: 
                self._check_refit_for_multimetric(firset_test_score)
                refit_metric = self.refit
            
        if self.refit or not self.multimetric_: 
            self.best_index_ = self._select_best_index(
                self.refit, refit_metric, results
            )
            if not callable(self.refit): 
                self.best_score_ = results[f"mean_test_{refit_metric}"][
                    self.best_index_
                ]
            self.best_params_ = results["params"][self.best_index_]
            
        if self.refit: 
            self.best_estimator_ = clone(
                clone(base_estimator).set_params(**self.best_params_)
            )
            refit_start_time = time.time()
            if y is not None: 
                self.best_estimator_.fit(X, y, X2, **fit_params)
            else: 
                self.best_estimator_.fit(X, X2, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time
            
            if hasattr(self.best_estimator_, "feature_names_in_"): 
                self.feature_names_in_ = self.best_estimator_.feature_names_in
        
        self.scorer_ = scorers
        
        self.cv_results_ = results
        self.n_splits_ = n_splits
        
        return self
    
    
    def predict(self, X, X2): 
        check_is_fitted(self)
        return self.best_estimator_.predict(X, X2)
    
    def score(self, X, y, X2): 
        _check_refit(self, "score")
        check_is_fitted(self)
        if self.scorer_ is None:
            raise ValueError(
                "No score function explicitly defined, "
                "and the estimator doesn't provide one %s"
                % self.best_estimator_
            )
        if isinstance(self.scorer_, dict):
            if self.multimetric_:
                scorer = self.scorer_[self.refit]
            else:
                scorer = self.scorer_
            return scorer(self.best_estimator_, X, y, X2)

        # callable
        score = self.scorer_(self.best_estimator_, X, y, X2)
        if self.multimetric_:
            score = score[self.refit]
        return score
        