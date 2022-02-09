from typing import Callable, Tuple
try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
import lmfit
import pandas as pd
import models

def glm_error_function(
        parameters: lmfit.Parameters,
        data: np.ndarray,
        objective_function: Callable,
        regressor_df: pd.DataFrame,
        args: dict = None) -> np.ndarray:
    """
    Parameters
    ----------
    parameters : lmfit.Parameters
        A dictionary of values representing a model setting.
    data : np.ndarray [1D]
       The actual, measured time-series against which the model is fit.
    objective_function : callable
        The objective function that takes `parameters` and `args` and
        produces a model time-series.
    regressor_df : pd.DataFrame
        DataFrame representing low-frequency drifts, intercept, and nuisances to be regressed out.
        the columns in the df link directly to the names of parameters in the parameters argument.
    args : dictionary
        A dictionary with additional parameters for the fitting routine. Default is None

    Returns
    -------
    error : float
        The residual sum of squared errors between the prediction and data.
    """
    prediction = objective_function(parameters, **args)
    regressor_df['glm_prf_amplitude'] = prediction
    betas, residuals, _, _ = np.linalg.lstsq(regressor_df, data, rcond=None)
    for reg_name, beta in zip(regressor_df.columns, betas):
        parameters[reg_name].value = beta

    # optional other option:
    # return residuals

    # get y-hat from GLM betas and dm, and return error "timecourse"
    model = np.dot(betas, regressor_df.T)
    return np.nan_to_num(model - data, nan=1e12).sum()

def diff_error_function(
        parameters: lmfit.Parameters,
        data: np.ndarray,
        objective_function: Callable,
        args: dict = None) -> np.ndarray:
    """
    Parameters
    ----------
    parameters : lmfit.Parameters
        A dictionary of values representing a model setting.
    data : np.ndarray [1D]
       The actual, measured time-series against which the model is fit.
    objective_function : callable
        The objective function that takes `parameters` and `args` and
        produces a model time-series.
    args : dictionary
        A dictionary with additional parameters for the fitting routine. Default is None

    Returns
    -------
    error : float
        The residual sum of squared errors between the prediction and data.
    """
    prediction = objective_function(parameters, **args)
    return np.nan_to_num(prediction - data, nan=1e12).sum()

@jit
def fit_glm(self,
             data: np.ndarray,
             design_matrix: pd.DataFrame) -> Tuple:
    """fit_glm performs fit for a single model prediction

    Parameters
    ----------
    data : np.ndarray
        data, 2D, last dimension time, first dimension units (voxels, whatever)
    design_matrix : pd.DataFrame
        design matrix for the GLM. Minimally this would be a pRF model prediction and an intercept regressor,
        but can also contain nuisances, etc etc.

    Returns
    -------
    Tuple (betas, rsq)

    """
    betas, _, _, _ = np.linalg.lstsq(design_matrix, data, rcond=None)
    model = np.dot(betas, design_matrix.T)
    rsq = np.sum((model-data)**2)/np.sum(data**2)
    return betas, rsq