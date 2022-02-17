from abc import ABC, abstractmethod
try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
    from numba import jit
import lmfit
from stimuli.stimulus import Stimulus
from signal.irf import IRF
from signal.filter import Filter

class Model(ABC):
    """Model

    Class that takes care of generating grids for pRF fitting and simulations
    """

    def __init__(self,
                 stimulus: Stimulus,
                 irf: IRF = None,
                 filter: Filter = None,
                 normalize_RFs: bool = True,
                 **kwargs) -> None:
        """__init__

        constructor for Model, takes stimulus, irf and filter objects as argument

        Parameters
        ----------
        stimulus : Stimulus
            Stimulus object containing information about the stimulus,
            and the space in which it lives.

        """
        pass

    @abstractmethod
    def return_prediction(self,
                          parameters: lmfit.Parameters) -> np.ndarray:
        """return_prediction returns a prediction timecourse given some set of lmfit parameters

        Parameters
        ----------
        parameters : lmfit.Parameters
            the Parameters dictionary for this model
        """
        pass

