import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import numba
import pandas as pd
import joblib

class PandasColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer class (https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html)
    Ported back from acceleration model code (https://github.com/SGLA-Data-Team/acceleration-transformations).
    """
    def __init__(self, func=None):
        """Construct a `CustomColumnTransformer` with `func` as the transformation function
            
            Parameters
            ----------
            func: a function, default `None`
                A function acceptabe by `pandas.DataFrame.apply()`

            Returns
            -------
            None

        """
        super().__init__()
        if func is None:
            self.func = lambda x: x  # identity transformation by default
        else:
            self.func = func

    def fit(self, X, y=None):
        """Unused. Maintained for compatibility purposes
        """
        return self

    def transform(self, X, *_):
        """Return transformation result
        """
        return X.apply(self.func, axis=1).to_frame('encoded')

class NumbaColumnTransformer(BaseEstimator, TransformerMixin):
    """
    A faster version of `CustomColumnTransformer`. Uses Numba. Currently supports transformations of single column.
    
    """
    def __init__(self, func,func_arg = None):
        """Construct a `CustomColumnTransformer` with `func` as the transformation function
            
            Parameters
            ----------
            func: a function, default `None`
                A scalar function.
               
            func_arg: str
                Name of the column that is to be transformed

            Returns
            -------
            None

        """
        super().__init__()
        if func is None:
            self.func = lambda x: x  # identity transformation by default
        else:
            self.func = func
        numba_func = numba.jit(func,forceobj=True)
        self.func_arg = func_arg
        def apply_func(col_a):
            n = len(col_a)
            result = np.empty(n, dtype='float64')
            for i in range(n):
                result[i] = numba_func(col_a.values[i])
            return result
        self.apply_func = numba.jit(apply_func,forceobj=True)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X,*_):
        result = self.apply_func(X[self.func_arg])
        return pd.DataFrame(pd.Series(result, index=X.index, name='encoded'))
    
def encoder_numba(var):
    if var>10:
        return 1
    return 0
def encoder_pandas(row,col='var1'):
    if row['var1']>10:
        return 1
    return 0