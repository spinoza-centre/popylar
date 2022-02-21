from abc import ABC, abstractmethod
import numpy as np
from numba import jit
import lmfit

class IRF(ABC):
    """IRF

    Generic class for implementing impulse response functions, used for convolving model timecourses
    """
    # standard: no fittable parameters
    parameters: list = []

    @abstractmethod
    def convolve(self,
                 prediction: np.ndarray,
                 parameters: lmfit.Parameters) -> np.ndarray:
        pass

class Null_IRF(IRF):
    """Null_IRF creates a 'null' IRF, which doesn't convolve the prediction timeseries at all
    """
    def convolve(self,
                 prediction: np.ndarray,
                 parameters: lmfit.Parameters = None) -> np.ndarray:
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

class Arbitrary_IRF(IRF):
    """HRF class for arbitrary response functions, supplied as an numpy.ndarray
    """
    def __init__(self,
                irf_kernel: np.ndarray) -> None:

        """__init__ takes a model instance for which this HRF will be used

        Parameters
        ----------
        irf_kernel: np.ndarray
            the actual kernel to use

        """
        self.irf_kernel = irf_kernel

    @jit
    def convolve(self,
                 prediction: np.ndarray,
                 parameters: lmfit.Parameters) -> np.ndarray:
        """convolves the prediction with a custom IRFs:
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

        prediction = np.convolve(prediction,
                                    self.irf_kernel,
                                    mode='full')[:prediction.shape[0]]
        return prediction