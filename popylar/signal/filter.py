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
                 filter_parameters: dict):
        self.filter_parameters = filter_parameters

    def filter(self,
                prediction: np.ndarray,
                sample_rate: float,
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

        signal_duration = prediction.shape[-1] / sample_rate
        drop_highpass_modes = 1 + np.floor(2 * self.filter_parameters['highpass_freq'] * signal_duration)

        coeffs = sp.fft.dct(prediction, norm='ortho', axis=-1)
        coeffs[:, :drop_highpass_modes] = 0
        filtered_prediction = sp.fft.idct(coeffs, norm='ortho', axis=-1)

        if not 'highpass_add' in self.filter_parameters.keys():
            return filtered_prediction
        elif self.filter_parameters['highpass_add'] == 'median':
            return filtered_prediction + np.median(prediction, axis=-1)
        elif self.filter_parameters['highpass_add'] == 'mean':
            return filtered_prediction + np.mean(prediction, axis=-1)

class SG_Filter(Filter):
    """SG_Filter performs a savitzky-golay high-pass filter on a timeseries.
    """
    def __init__(self,
                 filter_parameters: dict):
        self.filter_parameters = filter_parameters

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
                                                    window_length=self.filter_parameters['highpass_sg_window_length'],
                                                    polyorder=self.filter_parameters['highpass_sg_polyorder'])

        if not 'highpass_add' in self.filter_parameters.keys():
            return filtered_prediction
        elif self.filter_parameters['highpass_add'] == 'median':
            return filtered_prediction + np.median(prediction, axis=-1)
        elif self.filter_parameters['highpass_add'] == 'mean':
            return filtered_prediction + np.mean(prediction, axis=-1)