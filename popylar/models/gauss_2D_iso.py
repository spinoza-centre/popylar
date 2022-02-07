try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
import lmfit

from model import Model
from stimuli.stimulus import PRFStimulus2D
from irf.irf import IRF, Null_IRF, SPM_HRF, SPM_DD_HRF

class Iso2DGaussianModel(Model):
    def __init__(self,
                 stimulus: PRFStimulus2D = None,
                 irf: IRF = None,
                 normalize_RFs: bool = True,
                 **kwargs):
        """__init__ for Iso2DGaussianModel

        constructor, sets up stimulus and hrf for this Model

        Parameters
        ----------
        stimulus : PRFStimulus2D
            Stimulus object specifying the information about the stimulus,
            and the space in which it lives.
        irf : IRF, optional
            IRF object specifying the shape of the IRF for this Model.
            (the default is None, which implements no IRF)
        normalize_RFs : whether or not to normalize the RF volumes (default is True).
        """
        if irf is not None:
            self.irf = irf
        else:
            self.irf = Null_IRF()
        self.normalize_RFs = normalize_RFs

    def return_prediction(self,
                          parameters: lmfit.Parameters) -> np.ndarray:
        """return_prediction returns a prediction timecourse given some set of lmfit parameters

        Parameters
        ----------
        parameters : lmfit.Parameters
            the Parameters dictionary for this model
        """
        rf = gauss2D_iso_cart(x=self.stimulus.masked_coordinates[0],
                              y=self.stimulus.masked_coordinates[1],
                              mu=[parameters['prf_x'].value, parameters['prf_y'].value],
                              sigma=parameters['prf_size'].value,
                              norm_sum=self.normalize_RFs)
        raw_tc = np.dot(rf, self.stimulus.masked_design_matrix)

        return self.irf.convolve(prediction=raw_tc,
                          parameters=parameters)