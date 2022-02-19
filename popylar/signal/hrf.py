try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
    from numba import jit
import lmfit
import nilearn.glm.first_level.hemodynamic_models as hemo
import models
from irf import IRF

class HRF(IRF):
    """HRF class for haemodynamic response functions
    """
    def __init__(self,
                model: models.Model,
                hrf_type: str = 'spm',
                time_length: float = 50.0):

        """__init__ takes a model instance for which this HRF will be used

        Parameters
        ----------
        model : models.Model
            the model instance which will provide the necessary parameters,
            such as TR, etc, etc.
        hrf_type : str
            the hrf type, of 'spm' or 'glover'. default is 'spm'
        time_length : float
            the length of the hrf kernel timecourse, in [s] as arg for nilearn's hemodynamic_models

        """
        self.model = model
        self.TR = 1.0 / self.model.stimulus.sample_rate
        self.time_length = time_length

        if hrf_type == 'spm':
            self.hrf_func = hemo.spm_hrf
        elif hrf_type == 'glover':
            self.hrf_func = hemo.glover_hrf

        self.hrf_kernel = self.hrf_func(self.TR,
                                    oversampling=1,
                                    time_length=self.time_length,
                                    onset=0.)

    def convolve(self,
                 prediction: np.ndarray,
                 parameters: lmfit.Parameters) -> np.ndarray:
        """convolves the prediction with a combination of three SPM HRFs:
        the original, the time-derivative, and the dispersion derivative.

        Parameters
        ----------
        prediction : np.ndarray
            prediction to convolve
        parameters : lmfit.Parameters, optional


        Returns
        -------
        np.ndarray
            the original prediction
        """

        hrf_prediction = np.convolve(prediction,
                                    self.hrf_kernel,
                                    mode='full')[:prediction.shape[0]]
        return prediction

class DD_HRF(HRF):
    """DD_HRF uses nilearn's HRF functions and their time- and dispersion derivatives,
    the gains of the latter two are taken from parameters and can be given or optimized
    """
    def __init__(self,
                model: models.Model,
                hrf_type: str = 'spm',
                time_length: float = 40.0):
        super().__init__(self, model=model, hrf_type=hrf_type, time_length=time_length)

        if hrf_type == 'spm':
            self.hrf_dt_func = hemo.spm_time_derivative
            self.hrf_dd_func = hemo.spm_dispersion_derivative
        elif hrf_type == 'glover':
            self.hrf_dt_func = hemo.glover_time_derivative
            self.hrf_dd_func = hemo.glover_dispersion_derivative
        else:
            raise ValueError(f'Unknown hrf_type {hrf_type}')

        self.hrf_dt_kernel = self.hrf_dt_func(self.TR,
                                            oversampling=1,
                                            time_length=self.time_length,
                                            onset=0.)
        self.hrf_dd_kernel = self.hrf_dd_func(self.TR,
                                            oversampling=1,
                                            time_length=self.time_length,
                                            onset=0.)
        # this HRF shape does have fittable HRF parameters,
        # these will be used by the Fitter object
        self.parameters = [lmfit.Parameter(name='hrf_td_gain',
                                           value=0.0,
                                           min=-50,
                                           max=50,
                                           vary=False),
                            lmfit.Parameter(name='hrf_dd_gain',
                                             value=0.0,
                                             min=-50,
                                             max=50,
                                             vary=False)]


    def convolve(self,
                 prediction: np.ndarray,
                 parameters: lmfit.Parameters) -> np.ndarray:
        """convolves the prediction with a combination of three SPM HRFs:
        the original, the time-derivative, and the dispersion derivative.

        Parameters
        ----------
        prediction : np.ndarray
            prediction to convolve
        parameters : lmfit.Parameters,
            must contain 'hrf_td_gain' and 'hrf_dd_gain' parameters


        Returns
        -------
        np.ndarray
            the prediction
        """
        assert 'hrf_td_gain' in parameters.keys(), \
                "need hrf_td_gain and hrf_dd_gain parameters to fit the DD_HRF shape"
        hrf_prediction = np.convolve(prediction,
                                    self.hrf_kernel,
                                    mode='full')[:prediction.shape[0]]
        hrf_td_prediction = np.convolve(prediction,
                                        self.hrf_dt_kernel,
                                        mode='full')[:prediction.shape[0]]
        hrf_dd_prediction = np.convolve(prediction,
                                        self.hrf_dd_kernel,
                                        mode='full')[:prediction.shape[0]]

        return prediction + parameters['hrf_td_gain'].value * hrf_td_prediction + parameters['hrf_dd_gain'].value * hrf_dd_prediction
