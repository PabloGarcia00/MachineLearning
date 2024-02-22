"""Module for emulating behavior of regsubset function in R from LEAPS library.
"""
from __future__ import annotations

from typing import Union, Optional, Iterable, Tuple, Dict, List
from itertools import combinations
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
import matplotlib.pyplot as plt

"""As seen in the docstring of the `exhaustive_search()` function, and also
from the type annotations of its parameters, the function takes in an `OLS`
or `ClassifierMixin` object as model.  The `OLS` object is also defined in
the `regusbsets()` module.  This class is a wrapper around the
`statsmodels.OLS` model to make it behave like a scikit-learn model and to
incorporate data subsetting.  `ClassifierMixin` is a parent class for all
classifiers from scikit-learn.

This wrapper was made because `statsmodels.OLS` has more descriptive output
than scikit-learn regressors, and is therefore more suitable for this lab. 
However, `statsmodels` does not contain classifiers (and scikit-learn does). 
We want our subset search functions be able to take in regressors and
classifiers in order to make the subset search functions model agnostic.  To
make this possible, we have to make the input models behave like each other. 
Of course, there are many other ways to make the subset search functions
model agnostic.
"""

def prepare_data(
    X: Union[np.array, pd.DataFrame],  # [N, M]
    subset: Optional[Iterable[int]] = None
) -> np.array:
    """Casts predictor data to np.array and subsets columns if supplying subset.

    Arg: X (np.array or pd.DataFrame): predictor data as np.array or pd.DataFrame.
    Arg: subset (iterable of ints, default: None): subsets columns at indices of ints from X. 
    Out: np.array: subsetted predictor data cast as np.array.
    """
    X = np.array(X)

    if subset is None:
        return X
    else:
        return X[:, subset]


class OLS(BaseEstimator, RegressorMixin):
    """Wraps sm.OLS as sklearn.base.RegressorMixin.
    """
    def __init__(self, fit_intercept=True) -> None:
        """Initializes model wrapper instance.

        Arg: fit_intercept (bool): add intercept during fitting if True.
        """
        self.model_class = sm.OLS
        self.fit_intercept = fit_intercept

    def __repr__(self) -> str:
        """Returns string representation of fitted model.

        Out: str: string representation of fitted model (e.g., coefficients).
        """
        if hasattr(self, 'fitted_'):
            if hasattr(self, 'feature_names'):
                return str(self.fitted_.summary(xname=self.feature_names))
            else:
                return str(self.fitted_.summary())
        else:
            raise ValueError('model is not yet fitted')

    def coefs(self) -> List[Tuple[str, float]]:
        """Returns model coefficients.

        Out: List[Tuple[str, float]]: list of tuples as (name: str, coef: float).
        """
        coefs = list(self.fitted_.params)
        if hasattr(self, 'feature_names'):
            return [i for i in zip(self.feature_names, coefs)]
        else:
            if self.fit_intercept:
                start = 0
            else:
                start = 1
                
            feature_names = [f'x{i}' for i in range(start, len(coefs))]

            if self.fit_intercept:
                feature_names[0] = 'intercept'

            return [i for i in zip(feature_names, coefs)]

    def fit(
        self, 
        X: Union[np.array, pd.DataFrame],  # [N, M] 
        y: Union[np.array, pd.Series],  # [N,]
        subset: Optional[Iterable[int]] = None
    ) -> OLS:
        """Fit a sm.OLS model with X and y.

        Arg: X (np.array or pd.DataFrame): predictor data as np.array or pd.DataFrame.
        Arg: y (np.array or pd.Series): target data as np.array or pd.Series.
        Arg: subset (iterable of ints, default: None): subsets columns at indices of ints from X. 
        Out: OLS: fitted sm.OLS model in OLS wrapper.
        """
        if subset is not None and isinstance(X, pd.DataFrame):
            self.feature_names = [list(X.columns)[i] for i in subset]
            if self.fit_intercept:
                self.feature_names = ['intercept', *self.feature_names]

        X = prepare_data(X, subset)

        if self.fit_intercept:
            X = sm.add_constant(X)

        self.model_ = self.model_class(y, X)
        self.fitted_ = self.model_.fit()

        return self
        
    def predict(
        self, 
        X: Union[np.array, pd.DataFrame],  # [N, M]
        subset: Optional[Iterable[int]] = None
    ) -> np.array:
        """Predict target values from predictor values X.

        Arg: X (np.array or pd.DataFrame): predictor data as np.array or pd.DataFrame.
        Arg: subset (iterable of ints, default: None): subsets columns at indices of ints from X. 
        Out: np.array: predicted target values by fitted wrapped sm.OLS model.
        """
        X = prepare_data(X, subset)

        if self.fit_intercept:
            X = sm.add_constant(X)

        return self.fitted_.predict(X)

    def evaluate(
        self, 
        X: Union[np.array, pd.DataFrame],  # [N, M]
        y_true: Union[np.array, pd.Series],  # [N,] 
        subset: Optional[Iterable[int]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluate predicted target values from predictor values X with true target values for X.

        Arg: X (np.array or pd.DataFrame): predictor data as np.array or pd.DataFrame.
        Arg: y_true (np.array or pd.Series): true target data for X as np.array or pd.Series.
        Arg: subset (iterable of ints, default: None): subsets columns at indices of ints from X. 
        Out: (float, dict as {str: float, ...}): calculated residual sum of squares, dictionary with
            calculated metrics of fitted model (key: str, value: float).
        """
        y_pred = self.predict(X, subset)

        rss = ((y_pred - y_true) ** 2).sum()

        metrics = {
            'n': len(subset),
            'rss': rss,
            'aic': self.fitted_.aic,
            'bic': self.fitted_.bic,
            'rsquared': self.fitted_.rsquared,
            'rsquared_adj': self.fitted_.rsquared_adj
        }

        return rss, metrics


class Result:
    """Stores results of a single best subset search. 
    """
    def __init__(self) -> None:
        """Initializes Result storage container.
        """
        self.ns = []
        self.metrics = defaultdict(list)
        self.subsets = []
        self.models = []

    def add_result(
        self, 
        new_metrics: Dict[str, float], 
        new_subset: Iterable[int], 
        new_model: Union[OLS, ClassifierMixin]
    ) -> None:
        """Add a new result to the result storage container.

        Arg: new_metrics (dict as {str: float, ...}): calculated residual sum of squares, 
            dictionary with calculated metrics of fitted model (key: str, value: float).
        Arg: new_subset (iterable of ints): subset used to fit model with. 
        Arg: new_model (OLS or ClassifierMixin): fitted model.
        """
        self.ns.append(len(new_subset))
        self.subsets.append(new_subset)
        self.models.append(new_model)

        for metric, score in new_metrics.items():
            self.metrics[metric].append(score)

    def get_model(self, n: int) -> Union[OLS, ClassifierMixin]:
        """Return model with feature subset size of `n`.

        Arg: n (int): feature subset size to return model for.
        Out: Out: Union[OLS, ClassifierMixin]: best model for feature subset size of `n`.
        """
        return self.models[n - 1]

    def best_model(
        self, 
        metric: str, 
        best: Union[min, max] = min
    ) -> Union[OLS, ClassifierMixin]:
        """Returns best model based on lowest (min) or highest (max) value of given metric.

        Arg: metric (str): metric to use for determining best model.
        Arg: best(min or max, default: min): use lowest (min) or highest (max) value of metric.
        Out: Union[OLS, ClassifierMixin]: best model based on given metric.
        """
        n = self.metrics[metric].index(best(self.metrics[metric]))

        return self.models[n]
    
    def plot(self) -> None:
        """Plot calculated metrics for best fitted model per predictor feature subset size.
        """
        for metric, scores in self.metrics.items():
            if len(scores) == len(self.ns):
                plt.plot(self.ns, scores)
                #DR
                if metric in ['rss','aic','bic']:
                  opt = scores.index(min(scores))
                else:
                  opt = scores.index(max(scores))
                plt.scatter(self.ns[opt],scores[opt],c='red')
                plt.xlabel('n')
                plt.ylabel(metric)
                plt.show()


def exhaustive_search(
    model: Union[OLS, ClassifierMixin], 
    X: Union[np.array, pd.DataFrame],  # [N, M]
    y: Union[np.array, pd.Series],  # [N,] 
    nvmax: int
) -> Result:
    """Perform exhaustive search for best subset of predictor features. 

    Arg: model (OLS or ClassifierMixin): model used for exhaustive search.
    Arg: X (np.array or pd.DataFrame): predictor data as np.array or pd.DataFrame.
    Arg: y (np.array or pd.Series): target data as np.array or pd.Series.
    Arg: nvmax (int, default: 4): calculate best feature subset for feature subset 
        sizes of `nvmax` and lower.
    Out: Result: storage container storing best fitted model with calculated metrics
        per predictor feature subset size. 
    """ 
    results = Result()

    for nfeat in range(1, nvmax + 1):
        print(f"calculating scores for 'nfeat={nfeat}'...")

        best_score = float('inf') 
        best_metrics = None
        best_subset = []
        best_model = None

        for subset in combinations(range(X.shape[1]), nfeat):
            fitted_model = model.fit(X, y, subset)
            score, metrics = model.evaluate(X, y, subset)

            if score <= best_score:
                best_score = score
                best_metrics = metrics
                best_subset = subset 
                best_model = deepcopy(fitted_model)
        
        results.add_result(best_metrics, best_subset, best_model)

    print('done')
    return results


def forward_search(
    model: Union[OLS, ClassifierMixin], 
    X: Union[np.array, pd.DataFrame],  # [N, M] 
    y: Union[np.array, pd.Series],  # [N,] 
    nvmax: int
) -> Result:
    """Perform forward search for best subset of predictor features. 

    Arg: model (OLS or ClassifierMixin): model used for exhaustive search.
    Arg: X (np.array or pd.DataFrame): predictor data as np.array or pd.DataFrame.
    Arg: y (np.array or pd.Series): target data as np.array or pd.Series.
    Arg: nvmax (int, default: 4): calculate best feature subset for feature subset 
        sizes of `nvmax` and lower.
    Out: Result: storage container storing best fitted model with calculated metrics
        per predictor feature subset size. 
    """ 
    results = Result()

    best_metrics = None
    best_subset = []
    best_model = None
#DR    best_score = float('inf')

    subset_options = [[ind] for ind in range(X.shape[1])]

    while len(best_subset) < nvmax:
        #DR
        best_score = float('inf')
        for subset in subset_options:
            fitted_model = model.fit(X, y, subset)
            score, metrics = model.evaluate(X, y, subset)

            if score <= best_score:
                best_score = score
                best_metrics = metrics
                best_subset = subset
                best_model = deepcopy(fitted_model)

        subset_options = [
            best_subset + [i] 
            for i in range(X.shape[1]) 
            if i not in best_subset
        ]

        results.add_result(best_metrics, best_subset, best_model)

    return results


def backward_search(
    model: Union[OLS, ClassifierMixin], 
    X: Union[np.array, pd.DataFrame],  # [N, M] 
    y: Union[np.array, pd.Series],  # [N,] 
    nvmin: int
) -> Result:
    """Perform backward search for best subset of predictor features. 

    Arg: model (OLS or ClassifierMixin): model used for exhaustive search.
    Arg: X (np.array or pd.DataFrame): predictor data as np.array or pd.DataFrame.
    Arg: y (np.array or pd.Series): target data as np.array or pd.Series.
    Arg: nvmax (int, default: 4): calculate best feature subset for feature subset 
        sizes of `nvmax` and lower.
    Out: Result: storage container storing best fitted model with calculated metrics
        per predictor feature subset size. 
    """ 
    results = Result()

    # We start with a model that includes all predictor features
    best_subset = [i for i in range(X.shape[1])]
    best_model = model.fit(X, y, best_subset)
    _, best_metrics = model.evaluate(X, y, best_subset)
    results.add_result(best_metrics, best_subset, best_model)

    while len(best_subset) > nvmin:
        temp_results = []

        for feat in best_subset:
            subset = [ind for ind in best_subset if ind is not feat]
            fitted_model = model.fit(X, y, subset)
            score, metrics = model.evaluate(X, y, subset)

            temp_results.append((score, metrics, subset, deepcopy(fitted_model)))

        _, best_metrics, best_subset, best_model = min(temp_results, key=lambda t: t[0])

        results.add_result(best_metrics, best_subset, best_model)

    # invert results to keep them ordered from smallest to largest subset
    results.ns = results.ns[::-1]
    results.subsets = results.subsets[::-1]
    results.models = results.models[::-1]

    for metric, score in results.metrics.items():
        results.metrics[metric] = results.metrics[metric][::-1]

    return results