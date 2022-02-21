from abc import ABC, abstractmethod
try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
    from numba import jit
import pandas as pd
import lmfit
from joblib import Parallel, delayed
from copy import copy
from popylar import models
from popylar.fitters.fit_utils import iterative_search
import popylar.signal.hrf as hrf


class Fitter(ABC):
    """Fitter

    Superclass for classes that implement the different fitting methods,
    for a given model. It contains 2D-data and leverages a Model object.

    data should be two-dimensional so that all bookkeeping with regard to voxels,
    electrodes, etc is done by the user. Generally, a Fitter class should implement
    both a `grid_fit` and an `interative_fit` method to be run in sequence.

    """

    @abstractmethod
    def __init__(self,
                 data: np.ndarray,
                 model: models.Model,
                 **kwargs) -> None:
        pass

    @abstractmethod
    def grid_fit(self,
                 regressor_df: pd.DataFrame = None,
                 **kwargs) -> None:
        pass

    @abstractmethod
    def iterative_fit(self,
                      rsq_threshold: float = 0.0,
                      n_jobs: int = -1,
                      optimizer: str = "",
                      optimizer_settings: dict = None,
                      parameters: list = None,
                      error_function_type: str = 'difference',
                      **kwargs) -> None:
        pass


class PRFFitter(Fitter):
    def __init__(self,
                 data: np.ndarray,
                 model: models.Model,
                 dtype: type = np.float32,
                 **kwargs) -> None:
        """__init__ sets up data and model

        Parameters
        ----------
        data : numpy.ndarray, 2D
            input data. First dimension units, Second dimension time
        model : models.Model
            Model object that provides the grid and iterative search
            predictions.
        """
        assert len(data.shape) == 2, \
            "input data should be two-dimensional, with first dimension units and second dimension time"

        self.data = data.astype(dtype)
        self.model = model
        self.dtype = dtype

        self.__dict__.update(kwargs)

        self.n_units = self.data.shape[0]
        self.n_timepoints = self.data.shape[-1]
        self.data_var = self.data.var(axis=-1)

    def grid_fit(self,
                 regressor_df: pd.DataFrame = None,
                 **kwargs) -> None:
        """grid_fit performs grid fitting of the entire dataset

        Parameters
        ----------
        regressor_df : pd.DataFrame
            pd.DataFrame of regressors, to be used for nuisances etc in the GLM

        """
        assert hasattr(self, 'grid_predictions'), 'Please create grid predictions before grid fit'

        # set up the rest of the regressors
        if regressor_df is None:
            regressor_df = pd.DataFrame(np.ones((self.grid_predictions.shape[-1], 2)),
                                        columns=['prf_baseline', 'prf_amplitude'])

        # fit the grid predictions
        self.best_fit_betas = pd.DataFame(np.zeros((self.data.shape[0], regressor_df.shape[0])),
                                          columns=regressor_df.columns)
        self.best_fit_rsqs = np.zeros((self.data.shape[0]))
        self.best_fit_models = np.zeros((self.data.shape[0]))
        for gi, gp in enumerate(self.grid_predictions):
            # fill in actual prediction as a regressor for prf amplitude
            regressor_df['prf_amplitude'] = gp
            betas, rsqs = fit_utils.fit_glm(self.data, design_matrix=np.array(regressor_df))
            improved_fits = rsqs > self.best_fit_rsqs
            self.best_fit_betas[improved_fits] = pd.DataFame(
                betas[improved_fits], columns=regressor_df.columns)
            self.best_fit_rsqs[improved_fits] = rsqs[improved_fits]
            self.best_fit_models[improved_fits] = gi

    def iterative_fit(self,
                      rsq_threshold: float = 0.0,
                      n_jobs: int = -1,
                      optimizer: str = "trust-constr",
                      optimizer_settings: dict = None,
                      parameters: list = None,
                      error_function_type: str = 'difference',
                      fit_hrf: bool = False,
                      verbose: int = 1,
                      **kwargs) -> None:
        """iterative_fit performs an iterative fit over the data

        Parameters
        ----------
        rsq_threshold : float, optional
            which units in the data to fit, by default 0.0
        n_jobs : int, optional
            nr of jobs to spawn for fitting, by default -1
        optimizer : str, optional
            lmfit optimizer, by default "trust-constr"
        optimizer_settings : dict, optional
            settings for the optimizer, such as {maxiter, xtol, ftol, etc}, by default None
        parameters : list, optional
            list of lmfit Parameters objects, one for each unit, by default None
        fit_hrf: bool, optional
            Whether to fit the HRF parameters, by default False. Requires a DD_HRF IRF object in the Model.
        error_function_type : str, optional, by default 'difference'
            whether to calculate errors by direct subtraction of model prediction and data ("difference"), or
            by performing a GLM fit ("GLM") and subtracting yhat from the data. This latter option allows for
            nuisance regression during the fitting process, for example.
        """

        # set up Parameters objects for each unit, with their starting points
        # either from Grid Fit or from direct injection
        if hasattr(self, 'grid_fit_results'):
            rsq_mask = self.grid_fit_results['rsq_grid'] > rsq_threshold
        else:
            rsq_mask = np.ones((self.n_units), dtype=bool)
        assert rsq_mask.sum(
        ) > 0, f'there are no units that satisfy rsq_grid > rsq_threshold ({rsq_threshold})'
        self.rsq_mask = rsq_mask
        if fit_hrf:
            assert isinstance(self.model.irf, hrf.DD_HRF), 'Fitting the HRF shape requires a DD_HRF IRF object in the Model.'

        # if no parameters are given explicitly,
        # need to construct sensible default parameters from grid_fit_results
        if parameters is None:
            assert hasattr(self, 'grid_fit_results'), \
                'need to run grid fitting before iterative fitting if parameters are not explicitly given'
            parameters = []
            for unit in np.arange(self.n_units):
                # set up parameters
                pars = copy(self.standard_parameters)
                # fill in parameter values from grid fit
                for par_name in self.standard_parameters.valuesdict():
                    pars[par_name].value = self.grid_fit_results[par_name].iloc[unit]
                for irf_param in self.model.irf.parameters:
                    pars[irf_param.name] = irf_param
                    if fit_hrf:
                        pars[irf_param.name].vary = True
                # set variation of parameters
                if error_function_type is 'difference':
                    for par in pars:
                        par.vary = True
                elif error_function_type is 'GLM':
                    for par in pars:
                        if par.name in ['prf_baseline', 'prf_amplitude']:
                            par.vary = False
                        else:
                            par.vary = True
                parameters.append(pars)
        else:
            self.parameters = parameters

        self.minimizer_results = Parallel(n_jobs, verbose=verbose)(
            delayed(iterative_search)(model=self.model,
                                                data=data,
                                                parameters=params,
                                                optimizer=optimizer,
                                                optimizer_settings=optimizer_settings,
                                                error_function_type=error_function_type,
                                                verbose=verbose)
            for (data, params) in zip(self.data[self.rsq_mask], parameters[self.rsq_mask]))

        it_results = np.zeros((self.n_units, len(self.standard_parameters)))
        it_results[self.rsq_mask] = np.array(
            [[par.value for par in mr.params] for mr in self.minimizer_results])
        self.iterative_fit_results = pd.DataFrame(it_results,
                                                  columns=[param.name for param in self.standard_parameters])



