from abc import ABC, abstractmethod
try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
import scipy as sp
import lmfit
import nilearn.glm.first_level.hemodynamic_models as hemo
import models

class Filter(ABC):
    """Filter

    Generic class for implementing impulse response functions, used for convolving model timecourses
    """

    @abstractmethod
    def filter(self,
                prediction: np.ndarray,
                **kwargs = None) -> np.ndarray:
        pass

class Null_Filter(Filter):
    """Null_Filter creates a 'null' Filter, which doesn't filter the prediction timeseries at all
    """
    def filter(self,
                prediction: np.ndarray,
                **kwargs = None) -> np.ndarray:
        """convolve just returns the prediction

        Parameters
        ----------
        prediction : np.ndarray
            which gets returned immediately
        parameters : lmfit.Parameters, optional
            will not be consulted, by default None

        Returns
        -------
        np.ndarray
            the original prediction
        """
        return prediction

class DCT_Filter(Filter):
    """DCT_Filter performs a discrete cosine high-pass filter on a timeseries.
    """
    def __init__(self,
                 model: models.Model):
        self.model = model
        signal_duration = self.model.stimulus.masked_design_matrix.shape[-1] / self.model.stimulus.sample_rate
        self.drop_highpass_modes = 1 + np.floor(2 * self.model.filter_parameters['highpass_dct_freq'] * signal_duration)


    def filter(self,
                prediction: np.ndarray,
                **kwargs = None) -> np.ndarray:
        """filter just returns the prediction after throwing out certain DCT modes

        Parameters
        ----------
        prediction : np.ndarray
            which gets filtered

        Returns
        -------
        np.ndarray
            the filtered prediction
        """


        coeffs = sp.fft.dct(predictions, norm='ortho', axis=-1)
        coeffs[:, :self.drop_highpass_modes] = 0
        filtered_prediction = sp.fft.idct(coeffs, norm='ortho', axis=-1)

        if self.model.filter_parameters['highpass_add'] == 'median':
            return filtered_prediction + np.median(predictions, axis=-1)
        elif self.model.filter_parameters['highpass_add'] == 'mean':
            return filtered_prediction + np.mean(predictions, axis=-1)
        else:
            return filtered_prediction

class SG_Filter(Filter):
    """SG_Filter performs a savitzky-golay high-pass filter on a timeseries.
    """
    def __init__(self,
                 model: models.Model):
        self.model = model
        signal_duration = self.model.stimulus.masked_design_matrix.shape[-1] / self.model.stimulus.sample_rate
        self.polyorder = self.model.filter_parameters['highpass_sg_polyorder']
        self.window_length = self.model.filter_parameters['highpass_sg_window_length']


    def filter(self,
                prediction: np.ndarray,
                **kwargs = None) -> np.ndarray:
        """filter just returns the prediction after throwing out certain DCT modes

        Parameters
        ----------
        prediction : np.ndarray
            which gets filtered

        Returns
        -------
        np.ndarray
            the filtered prediction (or data)
        """


        filtered_prediction = sp.signal.savgol_filter(prediction,
                                                        window_length=self.window_length,
                                                        polyorder=self.polyorder)

        if self.model.filter_parameters['highpass_add'] == 'median':
            return predictions - filtered_prediction + np.median(predictions, axis=-1)
        elif self.model.filter_parameters['highpass_add'] == 'mean':
            return predictions - filtered_prediction + np.mean(predictions, axis=-1)
        else:
            return predictions - filtered_prediction