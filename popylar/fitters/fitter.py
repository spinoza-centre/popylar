from abc import ABC, abstractmethod
try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
import pandas as pd
import lmfit
from copy import copy
import models
import fit_utils

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
                      **kwargs) -> None:
        pass


class PRFFitter(Fitter):
    def __init__(self,
                 data: np.ndarray,
                 model: models.Model,
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

        self.data = data.astype('float32')
        self.model = model

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
            regressor_df = pd.DataFrame(np.ones(self.grid_predictions.shape[-1]), columns=['glm_prf_baseline'])

        # fit the grid predictions
        self.best_fit_betas = np.zeros((self.data.shape[0], regressor_df.shape[0]))
        self.best_fit_rsqs = np.zeros((self.data.shape[0]))
        self.best_fit_models = np.zeros((self.data.shape[0]))
        for gi, gp in enumerate(self.grid_predictions):
            regressor_df['glm_prf_amplitude'] = gp
            betas, rsqs = fit_utils.fit_glm(self.data, design_matrix=regressor_df)
            improved_fits = rsqs > self.best_fit_rsqs
            self.best_fit_betas[improved_fits] = betas[improved_fits]
            self.best_fit_rsqs[improved_fits] = rsqs[improved_fits]
            self.best_fit_models[improved_fits] = gi


class Iso2DGaussianFitter(PRFFitter):
    def __init__(self,
                 data: np.ndarray,
                 model: models.Model,
                 bounds: dict = None,
                 **kwargs) -> None:
        super().__init__(data: np.ndarray,
                         model: models.Model,
                         **kwargs)
        # standard values for bounds should come from a config yaml, but are hard-coded now
        # to be broad and conform to (more or less) the size of the visual field
        if bounds is None:
            bounds = dict(['prf_x', 'prf_y', 'prf_size'], [[-120,120], [-120,120], [0.05, 90]])
        self.standard_parameters = lmfit.Parameters()
        for par_name in ['prf_x', 'prf_y', 'prf_size']:
            self.standard_parameters[par_name] = lmfit.Parameter(name=par_name,
                                                            value=0.0,
                                                            min=bounds[par_name][0],
                                                            max=bounds[par_name][1])

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
        self.par_names = ['prf_x', 'prf_y', 'prf_size']
        # create predictions
        self.grid_predictions = np.zeros((self.grid_xs.shape[0], self.model.stimulus.masked_design_matrix.shape[-1]))
        for x, y, size, gi in zip(self.grid_xs, self.grid_ys, self.grid_sizes, np.arange(self.grid_xs.shape[0])):
            self.standard_parameters['prf_x'].value = x
            self.standard_parameters['prf_y'].value = y
            self.standard_parameters['prf_size'].value = size
            self.grid_predictions[gi] = self.model.return_prediction(self.standard_parameters)

    def collect_grid_results(self,
                             columns: list):
        """collect_grid_results takes earlier grid results and packages them into self.grid_fit_results

        Parameters
        ----------
        columns : list of str
            names of the columns of the grid design matrix
        """
        assert hasattr(self, 'best_fit_betas'), 'Please perform grid fit before collecting its results'
        # set up results
        self.grid_fit_results = pd.DataFrame(self.best_fit_betas, columns=columns)
        for parname, parvalues in zip(['prf_x', 'prf_y', 'prf_size'],
                                            [self.grid_xs,
                                            self.grid_ys,
                                            self.grid_sizes]):
            self.grid_fit_results[parname] = parvalues[self.best_fit_models]
        self.grid_fit_results['grid_rsq'] = self.best_fit_rsqs

    def grid_fit(self,
                 regressor_df: pd.DataFrame = None,
                 **kwargs) -> None:
        super.grid_fit(regressor_df: pd.DataFrame = None,
                 **kwargs)
        self.collect_grid_results(columns=regressor_df.columns)

    def iterative_fit(self,
                      rsq_threshold: float = 0.0,
                      n_jobs: int = -1,
                      optimizer: str = "trust-constr",
                      optimizer_settings: dict = None,
                      parameters: list = None,
                      **kwargs) -> None:
        """ iterative_fit generic function for iterative fitting.

        Parameters
        ----------
        rsq_threshold : float
            Grid-Fit rsq threshold for iterative fitting. Must be between 0 and 1.
        n_jobs : int, optional
            number of jobs to use for the computation. The default is -1.
        optimizer : str, optional
            Which optimizer to use for fitting. The default is trust-constr.
        optimizer_settings : dict, optional
            Optimizer-specific settings such as tolerances, max fun evals, etc.
            The default is None.
        parameters : list of lmfit.Parameters, optional
            parameters for each unit - can be injected from outside.
            The default is None, in which case the Fitter tries to get them from the grid stage
            In this case, the basic rule is that if the name of the parameter starts with `prf_`,
            it is set to vary

        Returns
        -------
        None.

        """

        # set up Parameters objects for each unit, with their starting points
        # either from Grid Fit or from direct injection
        if hasattr(self, grid_fit_results):
            rsq_mask = self.grid_fit_results['rsq_grid'] > rsq_threshold
        else:
            rsq_mask = np.ones((self.n_units), dtype=bool)
        if parameters is None:
            assert hasattr(self, grid_fit_results), \
                'need to run grid fitting before iterative fitting if no parameters are given.'
            parameters = []
            for unit in np.arange(self.n_units):
                pars = copy(self.standard_parameters)
                pars['prf_x'].value = self.grid_fit_results['prf_x'].iloc[unit]
                pars['prf_y'].value = self.grid_fit_results['prf_y'].iloc[unit]
                pars['prf_size'].value = self.grid_fit_results['prf_size'].iloc[unit]
                parameters.append(pars)




