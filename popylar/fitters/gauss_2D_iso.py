from abc import ABC, abstractmethod
import numpy as np
from numba import jit
import pandas as pd
import lmfit
from copy import copy

from popylar import models
from popylar.fitters import fit_utils
from popylar.fitters.fitter import PRFFitter

class Iso2DGaussianFitter(PRFFitter):
    def __init__(self,
                 data: np.ndarray,
                 model: models.Model,
                 bounds: dict = None,
                 **kwargs) -> None:
        super().__init__(data=data,
                         model=model,
                         **kwargs)
        # cannot assert a specific Iso2DGaussianModel instance here
        # because subclasses will need different models.
        assert isinstance(model, models.Model), \
            "Iso2DGaussianFitter requires a models.Iso2DGaussianModel instance as model"

        # standard values for bounds should come from a config yaml, these standards are hard-coded
        # to be broad and conform to (more or less) the size of the visual field
        if bounds is None:
            self.bounds = dict(zip(['prf_x', 'prf_y', 'prf_size', 'prf_baseline', 'prf_amplitude'], [
                          [-180, 180], [-180, 180], [0.05, 180], [-np.inf, np.inf], [-np.inf, np.inf]]))
        else:
            self.bounds = bounds

        self.standard_parameters = lmfit.Parameters()
        for par_name in ['prf_x', 'prf_y', 'prf_size', 'prf_baseline', 'prf_amplitude']:
            self.standard_parameters[par_name] = lmfit.Parameter(name=par_name,
                                                                 value=0.0,
                                                                 min=self.bounds[par_name][0],
                                                                 max=self.bounds[par_name][1])

    def create_grid_predictions(self,
                                x_grid: np.ndarray,
                                y_grid: np.ndarray,
                                size_grid: np.ndarray) -> None:
        """create_grid_predictions sets up an array of grid predictions

        Parameters
        ----------
        x_grid : np.ndarray, optional
            full grid of parameter settings in x
        y_grid : np.ndarray, optional
            full grid of parameter settings in y
        size_grid : np.ndarray, optional
            full grid of parameter settings in size
        """
        # set up separate variables
        grid_xs, grid_ys, grid_sizes = np.meshgrid(x_grid, y_grid, size_grid)
        self.grid_xs, self.grid_ys, self.grid_sizes = grid_xs.ravel(), grid_ys.ravel(), grid_sizes.ravel()
        self.par_names = ['prf_x', 'prf_y', 'prf_size', 'prf_baseline', 'prf_amplitude']
        # create predictions
        self.grid_predictions = np.zeros(
            (self.grid_xs.shape[0], self.model.stimulus.masked_design_matrix.shape[-1]), dtype=self.dtype)
        for x, y, size, gi in zip(self.grid_xs, self.grid_ys, self.grid_sizes, np.arange(self.grid_xs.shape[0])):
            self.standard_parameters['prf_x'].value = x
            self.standard_parameters['prf_y'].value = y
            self.standard_parameters['prf_size'].value = size
            self.standard_parameters['prf_baseline'].value = 0
            self.standard_parameters['prf_amplitude'].value = 1
            self.grid_predictions[gi] = self.model.return_prediction(self.standard_parameters)

    def collect_grid_results(self,
                             columns: list):
        """collect_grid_results takes earlier grid results and packages them into self.grid_fit_results

        Parameters
        ----------
        columns : list of str
            names of the columns of the grid design matrix
        """
        assert hasattr(
            self, 'best_fit_betas'), 'Please perform grid fit before collecting its results'
        # set up results
        self.grid_fit_results = self.best_fit_betas
        for parname, parvalues in zip(['prf_x', 'prf_y', 'prf_size'],
                                      [self.grid_xs,
                                       self.grid_ys,
                                       self.grid_sizes]):
            self.grid_fit_results[parname] = parvalues[self.best_fit_models]
        self.grid_fit_results['grid_rsq'] = self.best_fit_rsqs

    def grid_fit(self,
                 regressor_df: pd.DataFrame = None,
                 **kwargs) -> None:
        super().grid_fit(regressor_df=regressor_df,
                       **kwargs)
        self.collect_grid_results(columns=regressor_df.columns)


class CSSIso2DGaussianFitter(Iso2DGaussianFitter):
    def __init__(self,
                 data: np.ndarray,
                 model: models.Model,
                 bounds: dict = None,
                 **kwargs) -> None:
        super().__init__(data=data,
                         model=model,
                         bounds=bounds,
                         **kwargs)
        # cannot assert a specific Iso2DGaussianModel instance here
        # because subclasses will need different models.
        assert isinstance(self.model, models.CSSIso2DGaussianModel), \
            "CSSIso2DGaussianFitter requires a models.CSSIso2DGaussianModel instance as model"

        # standard values for bounds should come from a config yaml, these standards are hard-coded
        # to be broad and conform to (more or less) the size of the visual field
        self.standard_parameters['prf_css_exponent'] = lmfit.Parameter(name='prf_css_exponent',
                                                            value=1.0,
                                                            min=1e-9,
                                                            max=5,
                                                            vary=True)

    @classmethod
    def from_Iso2DGaussianFitter(linear_gauss:Iso2DGaussianFitter,
                                 model: models.CSSIso2DGaussianModel,
                                 **kwargs):
        """from_Iso2DGaussianFitter creates a new CSSIso2DGaussianFitter instance
         from a Iso2DGaussianFitter instance

        Parameters
        ----------
        linear_gauss : Iso2DGaussianFitter
            the to-be-internalized Fitter object
        """


        assert isinstance(self.model, models.CSSIso2DGaussianModel), \
            "CSSIso2DGaussianFitter requires a models.CSSIso2DGaussianModel instance as model"
        new_obj = CSSIso2DGaussianFitter.__init__(data=linear_gauss.data,
                                        model=model,
                                        bounds=linear_gauss.bounds,
                                        **kwargs)

        # now, internalize the linear gauss fitter's outcomes.
        new_obj.grid_fit_results = linear_gauss.grid_fit_results
        new_obj.linear_iterative_fit_results = linear_gauss.iterative_fit_results
        new_obj.linear_minimizer_results = linear_gauss.minimizer_results
        new_obj.parameters = []
        rsq_mask_units = np.arange(n_units)[rsq_mask]
        for unit in np.arange(new_obj.n_units):
            if unit in rsq_mask_units:
                params = copy(linear_gauss.minimizer_results[unit].params)
                params['prf_css_exponent'] = new_obj.standard_parameters['prf_css_exponent']
            else:
                params = copy(new_obj.standard_parameters)
            new_obj.parameters.append(params)

        return new_obj

    def iterative_fit(self,
                      rsq_threshold: float = 0.0,
                      n_jobs: int = -1,
                      optimizer: str = "trust-constr",
                      optimizer_settings: dict = None,
                      error_function_type: str = 'difference',
                      verbose: int = 1,
                      **kwargs) -> None:
        """iterative_fit _summary_

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
        error_function_type : str, optional, by default 'difference'
            whether to calculate errors by direct subtraction of model prediction and data ("difference"), or
            by performing a GLM fit ("GLM") and subtracting yhat from the data. This latter option allows for
            nuisance regression during the fitting process, for example.
        """
        super.iterative_fit(rsq_threshold=rsq_threshold,
                            n_jobs=n_jobs,
                            optimizer=optimizer,
                            optimizer_settings=optimizer_settings,
                            parameters=self.parameters,
                            error_function_type=error_function_type,
                            verbose=verbose,
                            kwargs=kwargs)