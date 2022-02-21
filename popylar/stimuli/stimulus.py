import numpy as np
from numba import jit

class Stimulus:
    """Stimulus

    Minimal pRF stimulus class, base of others
    Design matrix has final dimension time.

    """

    def __init__(self,
                 design_matrix: np.ndarray,
                 sample_rate: float,
                 coordinates: np.ndarray,
                 **kwargs):
        """__init__ initialize Stimulus

        Parameters
        ----------
        design_matrix : numpy.ndarray
            numpy array containing the design matrix, in [horizontal, vertical, time] dimensions
        sample_rate : float
            amount of samples per second
        coordinates : numpy.ndarray
            coordinates for the 'pixels' in each of the 'feature' dimensions of the design matrix
        """
        self.coordinates = [cd.astype(np.float32) for cd in coordinates]
        if design_matrix.dtype in (bool, np.uint8):
            self.design_matrix = design_matrix
        else:
            self.design_matrix = design_matrix.astype(np.float32)
        self.sample_rate = sample_rate
        self.n_timepoints = self.design_matrix.shape[-1]
        self.__dict__.update(kwargs)
        self.mask_dm()

    def mask_dm(self):
        """mask_dm sets up coordinates such that there are no coordinates for design matrix elements that are empty
        which also converts the design matrix to a 1D array for ease of implementation later
        """
        self.dm_mask = self.design_matrix.std(-1) != 0
        self.masked_design_matrix = self.design_matrix[self.dm_mask]
        self.masked_coordinates = [c[self.dm_mask] for c in self.coordinates]

class PRFStimulus1D(Stimulus):
    """PRFStimulus1D

    Minimal 1-dimensional pRF stimulus class,
    which takes an input design matrix and sets up its real-world dimensions.

    this type of stimulus could be an auditory pRF stimulus, a numerosity stimulus,
    or a connective field stimulus

    """
    def __init__(self,
                 design_matrix: np.ndarray,
                 sample_rate: float,
                 coordinates: np.ndarray,
                 **kwargs):
        """__init__ initialize PRFStimulus1D

        Parameters
        ----------
        design_matrix : numpy.ndarray
            numpy array containing the design matrix, in
            {frequency, log(frequency), numerosity, log(numerosity), distance} dimensions
        sample_rate : float
            amount of samples per second
        coordinates : numpy.ndarray
            [description]
        """
        super().__init__(design_matrix=design_matrix,
                         sample_rate=sample_rate,
                         coordinates=coordinates,
                         kwargs=kwargs)

class PRFStimulus2D(Stimulus):
    """PRFStimulus2D

    Minimal visual 2-dimensional pRF stimulus class,
    which takes an input design matrix and sets up its real-world dimensions.

    """
    def __init__(self,
                 design_matrix: np.ndarray,
                 sample_rate: float,
                 screen_size_cm: float,
                 screen_distance_cm: float,
                 **kwargs):
        """__init__ initialize PRFStimulus2D

        Parameters
        ----------
        design_matrix : numpy.ndarray
            numpy array containing the design matrix, in [horizontal, vertical, time] dimensions
        sample_rate : float
            amount of samples per second
        screen_size_cm : float
            the size of the screen in cms. This refers to the screen width, the first dimension of the design matrix.
        screen_distance_cm : float
            the distance from eye to screen in cms.
        """
        self.screen_size_cm = screen_size_cm
        self.screen_distance_cm = screen_distance_cm
        self.screen_size_degrees = 2.0 * \
            np.degrees(np.arctan(self.screen_size_cm /
                                 (2.0*self.screen_distance_cm)))
        width_coordinates = np.linspace(-self.screen_size_degrees/2,
                                self.screen_size_degrees/2,
                                design_matrix.shape[0],
                                endpoint=True)
        hw_ratio = design_matrix.shape[1]/design_matrix.shape[0]
        height_coordinates = np.linspace(-hw_ratio*self.screen_size_degrees/2,
                                hw_ratio*self.screen_size_degrees/2,
                                design_matrix.shape[1],
                                endpoint=True)
        self.x, self.y = np.meshgrid(width_coordinates, height_coordinates)
        super().__init__(design_matrix=design_matrix,
                         sample_rate=sample_rate,
                         coordinates=[self.x.T, self.y.T],
                         kwargs=kwargs)




