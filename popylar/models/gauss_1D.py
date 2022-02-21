import numpy as np
from numba import jit
import lmfit

from popylar.models import Model
from popylar.stimuli.stimulus import PRFStimulus1D
from popylar.signal.irf import IRF, Null_IRF
from popylar.signal.filter import Filter, Null_Filter

from popylar.rf import gauss1D_cart


class Iso1DGaussianModel(Model):
    def __init__(self,
                 stimulus: PRFStimulus1D = None,
                 irf: IRF = None,
                 filter: Filter = None,
                 normalize_RFs: bool = True,
                 **kwargs):
        """__init__ for Iso1DGaussianModel

        constructor, sets up stimulus and hrf for this Model

        Parameters
        ----------
        stimulus : PRFStimulus1D
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
        params = parameters.valuesdict()
        rf = gauss1D_cart(x=self.stimulus.masked_coordinates[0],
                                       y=self.stimulus.masked_coordinates[1],
                                       mu=params['prf_x'],
                                       sigma=params['prf_size'],
                                       norm_sum=self.normalize_RFs)

        raw_tc = np.dot(rf, self.stimulus.masked_design_matrix)
        conv_tc = self.irf.convolve(prediction=raw_tc,
                                    parameters=parameters)
        hp_conv_tc = self.filter.filter(conv_tc)
        final_tc = hp_conv_tc * params['prf_amplitude'] + \
            params['prf_baseline']
        return final_tc


class CSSIso1DGaussianModel(Model):
    """CSSIso1DGaussianModel implements the Compressive Spatial Summation model of Kay et al, 2013.
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
        params = parameters.valuesdict()
        rf = gauss1D_cart(x=self.stimulus.masked_coordinates[0],
                                       y=self.stimulus.masked_coordinates[1],
                                       mu=params['prf_x'],
                                       sigma=params['prf_size'],
                                       norm_sum=self.normalize_RFs)

        raw_tc = np.dot(rf, self.stimulus.masked_design_matrix)
        # To be clear:
        # we need to raise the 'neural' timecourse,
        # and NOT the spatial RF to the exponent
        css_tc = raw_tc ** params['prf_css_exponent']
        conv_tc = self.irf.convolve(prediction=css_tc,
                                    parameters=parameters)
        hp_conv_tc = self.filter.filter(conv_tc)
        final_tc = hp_conv_tc * params['prf_amplitude'] + \
            params['prf_baseline']
        return final_tc


class DoGIso1DGaussianModel(Model):
    """DoGIso1DGaussianModel implements the Difference of Gaussians model of Zuiderbaan et al, 2013.
    """
    @jit
    def return_prediction(self,
                          parameters: lmfit.Parameters) -> np.ndarray:
        """return_prediction returns a prediction timecourse given some set of lmfit parameters
        When fitting this model with a GLM error function, set the 'prf_amplitude' and 'prf_baseline'
        to 1 and 0, respectively, and don't vary these parameters.

        Parameters
        ----------
        parameters : lmfit.Parameters
            the Parameters dictionary for this model
        """
        params = parameters.valuesdict()
        rf_center = gauss1D_cart(x=self.stimulus.masked_coordinates[0],
                                              y=self.stimulus.masked_coordinates[1],
                                              mu=params['prf_center_x'],
                                              sigma=params['prf_center_size'],
                                              norm_sum=self.normalize_RFs)
        rf_surround = gauss1D_cart(x=self.stimulus.masked_coordinates[0],
                                                y=self.stimulus.masked_coordinates[1],
                                                mu=params['prf_surround_x'],
                                                sigma=params['prf_surround_size'],
                                                norm_sum=self.normalize_RFs)
        # DoG is, in essence, a linear model.
        # This means that we can create a single spatial receptive field
        # and dot this with the stimulus.
        rf = rf_center * params['prf_center_amplitude'] - \
            rf_surround * params['prf_surround_amplitude']

        raw_tc = np.dot(rf, self.stimulus.masked_design_matrix)
        conv_tc = self.irf.convolve(prediction=raw_tc,
                                    parameters=parameters)
        hp_conv_tc = self.filter.filter(conv_tc)
        final_tc = params['prf_amplitude'] * hp_conv_tc + params['prf_baseline']
        return final_tc

class DNIso1DGaussianModel(Model):
    """DNIso1DGaussianModel implements the Divisive Normalization model of Aqil et al, 2021.
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
        params = parameters.valuesdict()
        rf_center = gauss1D_cart(x=self.stimulus.masked_coordinates[0],
                                              y=self.stimulus.masked_coordinates[1],
                                              mu=params['prf_center_x'],
                                              sigma=params['prf_center_size'],
                                              norm_sum=self.normalize_RFs)
        rf_surround = gauss1D_cart(x=self.stimulus.masked_coordinates[0],
                                                y=self.stimulus.masked_coordinates[1],
                                                mu=params['prf_surround_x'],
                                                sigma=params['prf_surround_size'],
                                                norm_sum=self.normalize_RFs)
        prf_tc = np.dot(rf_center, self.stimulus.masked_design_matrix)
        prf_surround_tc = np.dot(rf_surround, self.stimulus.masked_design_matrix)
        # since the DN model is a non-linear model like the CSS model,
        # we need to perform the division on the timecourses
        # and not their spatial profiles
        norm_tc = (params['prf_center_baseline'] + prf_tc * params['prf_center_amplitude']) / \
            (params['prf_surround_baseline'] + prf_surround_tc * params['prf_surround_amplitude'])
        norm_tc -= params['prf_center_baseline'] / params['prf_surround_baseline']

        conv_tc = self.irf.convolve(prediction=norm_tc,
                                    parameters=parameters)
        hp_conv_tc = self.filter.filter(conv_tc)
        final_tc = params['prf_amplitude'] * hp_conv_tc + params['prf_baseline']
        return final_tc
