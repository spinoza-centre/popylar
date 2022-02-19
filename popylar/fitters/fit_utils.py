from typing import Callable, Tuple
try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
    try:
        from numba import njit as jit
    except ImportError:
        print("need jax or numba for jit acceleration")
import lmfit
import pandas as pd
import models


@jit
def fit_glm(self,
            data: np.ndarray,
            design_matrix: np.ndarray) -> Tuple:
    """fit_glm performs fit for a single model prediction

    Parameters
    ----------
    data : np.ndarray
        data, 2D, last dimension time, first dimension units (voxels, whatever)
    design_matrix : np.ndarray
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


@jit
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
        DataFrame representing things like low-frequency drifts, intercept, and nuisances to be regressed out.
        the columns in the df link directly to the names of parameters in the parameters argument.
    args : dictionary
        A dictionary with additional parameters for the fitting routine. Default is None

    Returns
    -------
    error : float
        The residual sum of squared errors between the prediction and data.
    """
    # first dictate the baseline and amplitude of the prf
    parameters['prf_baseline'].value, parameters['prf_amplitude'].value = 0, 1
    prediction = objective_function(parameters, **args)
    regressor_df['prf_amplitude'] = prediction
    np_dm = np.array(regressor_df)

    betas, residuals, _, _ = np.linalg.lstsq(np_dm, data, rcond=None)
    for reg_name, beta in zip(regressor_df.columns, betas):
        parameters[reg_name].value = beta
    # optional other option:
    # return residuals

    # get y-hat from GLM betas and dm, and return error "timecourse"
    model = np.dot(betas, np_dm.T)
    return np.nan_to_num(model - data, nan=1e12)


@jit
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
    return np.nan_to_num(prediction - data, nan=1e12)


def iterative_search(model: models.Model,
                     data: np.ndarray,
                     parameters: lmfit.Parameters,
                     optimizer: str = "trust-constr",
                     optimizer_settings: dict = None,
                     error_function_type: str = 'difference',
                     verbose: int = 1,
                     regressor_df: pd.DataFrame = None,
                     **kwargs):
    """iterative_search performs the actual lmfit minimization

    Parameters
    ----------
    model : models.Model
        _description_
    data : np.ndarray [1D]
       The actual, measured time-series against which the model is fit.
    parameters : lmfit.Parameters
        A dictionary of values representing a model setting.
    optimizer : str, optional
        lmfit optimizer, by default "trust-constr"
    optimizer_settings : dict, optional
        settings for the optimizer, such as {maxiter, xtol, ftol, etc}, by default None
    error_function_type : str, optional, by default 'difference'
        whether to calculate errors by direct subtraction of model prediction and data ("difference"), or
        by performing a GLM fit ("GLM") and subtracting yhat from the data. This latter option allows for
        nuisance regression during the fitting process, for example.
    verbose : int, optional
        _description_, by default 1
    regressor_df : pd.DataFrame
        DataFrame representing things like low-frequency drifts, intercept, and nuisances to be regressed out.
        the columns in the df link directly to the names of parameters in the parameters argument.
    """

    if verbose > 0:
        print(
            f'Performing bounded, unconstrained minimization using {optimizer}, with parameters {parameters.pretty_print()}.')
    if error_function_type is 'difference':
        return lmfit.minimize(diff_error_function,
                              parameters,
                              method=optimizer,
                              args=(data, model.return_prediction),
                              fit_kws=optimizer_settings)
    elif error_function_type is 'GLM':
        return lmfit.minimize(glm_error_function,
                              parameters,
                              method=optimizer,
                              args=(data, model.return_prediction, regressor_df),
                              fit_kws=optimizer_settings)
