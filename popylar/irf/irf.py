from abc import ABC, abstractmethod
try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
import lmfit
import nilearn.glm.first_level.hemodynamic_models as hemo
import models

class IRF(ABC):
    """IRF

    Generic class for implementing impulse response functions, used for convolving model timecourses
    """

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

