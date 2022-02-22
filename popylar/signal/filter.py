from abc import ABC, abstractmethod
import numpy as np
from numba import jit
import scipy as sp
import lmfit
import nilearn.glm.first_level.hemodynamic_models as hemo
# from popylar import models

# @jit(nopython=True)
def remean(prediction: np.ndarray,
           filtered_prediction: np.ndarray,
           highpass_add: str = 'no') -> np.ndarray:
    if highpass_add == 'no':
        return filtered_prediction
    elif highpass_add == 'median':
        return (np.median(prediction, axis=-1) + filtered_prediction.T).T
    elif highpass_add == 'mean':
        return (np.mean(prediction, axis=-1) + filtered_prediction.T).T


class Filter(ABC):
    """Filter

    Generic class for high-pass filtering model timecourses.
    Will not likely be of much use, but is here for historical reasons.
    """

    @abstractmethod
    def filter(self,
                prediction: np.ndarray,
                **kwargs) -> np.ndarray:
        pass

class Null_Filter(Filter):
    """Null_Filter creates a 'null' Filter, which doesn't filter the prediction timeseries at all
    """
    def filter(self,
                prediction: np.ndarray,
                **kwargs) -> np.ndarray:
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
                n_timepoints: int,
                sample_rate: float,
                highpass_freq: float,
                highpass_add: str = 'no'):
        self.n_timepoints = n_timepoints
        self.sample_rate = sample_rate
        self.highpass_freq = highpass_freq
        self.highpass_add = highpass_add

        signal_duration = self.n_timepoints / self.sample_rate
        self.drop_highpass_modes = int(1 + np.floor(2 * self.highpass_freq * signal_duration))

    def filter(self,
                prediction: np.ndarray,
                **kwargs) -> np.ndarray:
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
        # if prediction.ndim == 1:
        #     prediction = prediction[np.newaxis, :]
        coeffs = sp.fft.dct(prediction, norm='ortho', axis=-1)
        coeffs[..., :self.drop_highpass_modes] = 0
        filtered_prediction = sp.fft.idct(coeffs, norm='ortho', axis=-1)

        return remean(prediction, filtered_prediction, highpass_add=self.highpass_add)

class SG_Filter(Filter):
    """SG_Filter performs a savitzky-golay high-pass filter on a timeseries.
    """
    def __init__(self,
                n_timepoints: int,
                window_length: int,
                polyorder: int = 3,
                highpass_add: str = 'no'):
        self.n_timepoints = n_timepoints
        assert window_length % 2 == 1, "sp.signal.savgol_filter window_length must be odd"
        self.window_length = window_length
        self.polyorder = polyorder
        self.highpass_add = highpass_add

    def filter(self,
                prediction: np.ndarray,
                **kwargs) -> np.ndarray:
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
        filtered_prediction = prediction - sp.signal.savgol_filter(prediction,
                                                    window_length=self.window_length,
                                                    polyorder=self.polyorder)

        return remean(prediction, filtered_prediction, highpass_add=self.highpass_add)
