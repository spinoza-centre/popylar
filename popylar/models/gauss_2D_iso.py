try:
    import jax.numpy as np
    from jax import jit
except ImportError:
    import numpy as np
    from numba import jit
import lmfit

from model import Model
from stimuli.stimulus import PRFStimulus2D
from signal.irf import IRF, Null_IRF
from signal.filter import Filter, Null_Filter

from rf import gauss2D_iso_cart

class Iso2DGaussianModel(Model):
    def __init__(self,
                 stimulus: PRFStimulus2D = None,
                 irf: IRF = None,
                 filter: Filter = None,
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
        filter : Filter, optional
            Filter object specifying the high-pass filter for this Model.
            (the default is None, which implements no filtering)
        normalize_RFs : whether or not to normalize the RF volumes (default is True).
        """
        self.irf = irf if irf is not None else Null_IRF()
        self.filter = filter if filter is not None else Null_Filter()

        self.normalize_RFs = normalize_RFs
        self.stimulus = stimulus

    @jit
    def return_prediction(self,
                          parameters: lmfit.Parameters) -> np.ndarray:
        """return_prediction returns a prediction timecourse given some set of lmfit parameters

        Parameters
        ----------
        parameters : lmfit.Parameters
            the Parameters dictionary for this model
        """
        rf = np.rot90(gauss2D_iso_cart(x=self.stimulus.masked_coordinates[0],
                              y=self.stimulus.masked_coordinates[1],
                              mu=[parameters['prf_x'].value, parameters['prf_y'].value],
                              sigma=parameters['prf_size'].value,
                              norm_sum=self.normalize_RFs))

        raw_tc = np.dot(rf, self.stimulus.masked_design_matrix)
        conv_tc = self.irf.convolve(prediction=raw_tc,
                                 parameters=parameters)
        hp_conv_tc = self.filter.filter(conv_tc)
        final_tc = hp_conv_tc * parameters['prf_amplitude'].value + parameters['prf_baseline'].value
        return final_tc

class CSSIso2DGaussianModel(Model):
    """CSSIso2DGaussianModel of Kay et al, 2013.
    """
    @jit
    def return_prediction(self,
                          parameters: lmfit.Parameters) -> np.ndarray:
        """return_prediction returns a prediction timecourse given some set of lmfit parameters

        Parameters
        ----------
        parameters : lmfit.Parameters
            the Parameters dictionary for this model
        """
        rf = np.rot90(gauss2D_iso_cart(x=self.stimulus.masked_coordinates[0],
                              y=self.stimulus.masked_coordinates[1],
                              mu=[parameters['prf_x'].value, parameters['prf_y'].value],
                              sigma=parameters['prf_size'].value,
                              norm_sum=self.normalize_RFs))

        raw_tc = np.dot(rf, self.stimulus.masked_design_matrix)
        css_tc = raw_tc ** parameters['prf_css_exponent']
        conv_tc = self.irf.convolve(prediction=css_tc,
                                 parameters=parameters)
        hp_conv_tc = self.filter.filter(conv_tc)
        final_tc = hp_conv_tc * parameters['prf_amplitude'].value + parameters['prf_baseline'].value
        return final_tc

class DoGIso2DGaussianModel(Model):
    """DoGIso2DGaussianModel of Zuiderbaan et al, 2013.
    """
    @jit
    def return_prediction(self,
                          parameters: lmfit.Parameters) -> np.ndarray:
        """return_prediction returns a prediction timecourse given some set of lmfit parameters

        Parameters
        ----------
        parameters : lmfit.Parameters
            the Parameters dictionary for this model
        """
        rf_center = np.rot90(gauss2D_iso_cart(x=self.stimulus.masked_coordinates[0],
                              y=self.stimulus.masked_coordinates[1],
                              mu=[parameters['prf_x'].value, parameters['prf_y'].value],
                              sigma=parameters['prf_size'].value,
                              norm_sum=self.normalize_RFs))
        rf_surround = np.rot90(gauss2D_iso_cart(x=self.stimulus.masked_coordinates[0],
                              y=self.stimulus.masked_coordinates[1],
                              mu=[parameters['prf_surround_x'].value, parameters['prf_surround_y'].value],
                              sigma=parameters['prf_surround_size'].value,
                              norm_sum=self.normalize_RFs))
        rf = rf_center * parameters['prf_amplitude'].value - rf_surround * parameters['prf_surround_amplitude'].value

        raw_tc = np.dot(rf, self.stimulus.masked_design_matrix)
        conv_tc = self.irf.convolve(prediction=raw_tc,
                                 parameters=parameters)
        hp_conv_tc = self.filter.filter(conv_tc)
        final_tc = hp_conv_tc + parameters['prf_baseline'].value
        return final_tc