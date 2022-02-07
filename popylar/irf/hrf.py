try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
import lmfit
import nilearn.glm.first_level.hemodynamic_models as hemo
import models
from irf import IRF

class HRF(IRF):
    """HRF class for haemodynamic response functions
    """
    def __init__(self,
                model: models.Model,
                time_length: float = 50.0):
        """__init__ takes a model instance for which this HRF will be used

        Parameters
        ----------
        model : models.Model
            the model instance which will provide the necessary parameters,
            such as TR, etc, etc.
        """
        self.model = model
        self.TR = 1.0 / self.model.stimulus.sample_rate
        self.time_length = time_length

class SPM_HRF(HRF):
    """SPM_HRF uses nilearn's spm HRF function
    """

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
                                    hemo.spm_hrf(self.TR,
                                    oversampling=1,
                                    time_length=self.time_length,
                                    onset=0.),
                                    mode='full')[:prediction.shape[0]]
        return prediction

class SPM_DD_HRF(HRF):
    """SPM_DD_HRF uses nilearn's spm HRF function and its time- and dispersion derivatives,
    the gains of the latter two are taken from parameters and can be given or optimized
    """

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
                                    hemo.spm_hrf(self.TR,
                                                oversampling=1,
                                                time_length=self.time_length,
                                                onset=0.),
                                    mode='full')[:prediction.shape[0]]
        hrf_td_prediction = np.convolve(prediction,
                                        hemo.spm_time_derivative(self.TR,
                                                                oversampling=1,
                                                                time_length=self.time_length,
                                                                onset=0.),
                                        mode='full')[:prediction.shape[0]]
        hrf_dd_prediction = np.convolve(prediction,
                                        hemo.spm_dispersion_derivative(self.TR,
                                                                        oversampling=1,
                                                                        time_length=self.time_length,
                                                                        onset=0.),
                                        mode='full')[:prediction.shape[0]]

        return prediction + parameters['hrf_td_gain'].value * hrf_td_prediction + parameters['hrf_dd_gain'].value * hrf_dd_prediction
