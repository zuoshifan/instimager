import abc
import numpy as np

import telescope
import Cylinder




class FourierTransformTelescope(telescope.TransitTelescope):
    """Common functionality for all Fourier Transform Telescopes.

    """

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class


    @property
    def k(self):
        """The central wavevector magnitude of each frequency band (in metres^-1)."""
        return 2 * np.pi / self.wavelengths




class UnpolarisedFourierTransformTelescope(FourierTransformTelescope, telescope.SimpleUnpolarisedTelescope):
    """A base for a unpolarised Fourier transform telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beam`.

    Abstract Methods
    ----------------
    beam : method
        Routines giving the field pattern for the feeds.
    """

    pass




class PolarisedFourierTransformTelescope(FourierTransformTelescope, telescope.SimplePolarisedTelescope):
    """A base for a polarised Fourier transform telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beamx` and `beamy`.

    Abstract Methods
    ----------------
    beamx, beamy : methods
        Routines giving the field pattern for the x and y feeds.
    """

    pass




class FFTTelescope(FourierTransformTelescope):
    """Common functionality for all Fast Fourier Transform Telescopes.

    """

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class

    auto_correlations = True


    @abc.abstractproperty
    def delta_u(self):
        """Spacing between two adjacent feeds in the u-direction, Unit: m."""
        return

    @abc.abstractproperty
    def delta_v(self):
        """Spacing between two adjacent feeds in the v-direction, Unit: m."""
        return

    @abc.abstractproperty
    def nfeeds_u(self):
        """Number of feeds in the u-direction."""
        return

    @abc.abstractproperty
    def nfeeds_v(self):
        """Number of feeds in the v-direction."""
        return

    @property
    def _single_feedpositions(self):
        """The set of feed positions.

        Returns
        -------
        feedpositions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """
        fplist = [ np.array([iu * self.delta_u, iv * self.delta_v]) for iu in range(self.nfeeds_u) for iv in range(self.nfeeds_v) ]

        return np.array(fplist)




class UnpolarisedFFTTelescope(FFTTelescope, UnpolarisedFourierTransformTelescope):
    """A base for a unpolarised Fast Fourier transform telescope.

    """

    pass




class PolarisedFFTTelescope(FFTTelescope, PolarisedFourierTransformTelescope):
    """A base for a polarised Fast Fourier transform telescope.

    """

    pass



class CylinderFFTTelescope(FFTTelescope, Cylinder.CylinderTelescope):
    """Cylinder type Fast Fourier Transform Telescopes.

    """

    in_cylinder = True
    non_commensurate = False


    @property
    def delta_u(self):
        """Spacing between two adjacent feeds in the u-direction, Unit: m."""
        return self.cylinder_spacing

    @property
    def delta_v(self):
        """Spacing between two adjacent feeds in the v-direction, Unit: m."""
        return self.feed_spacing

    @property
    def nfeeds_u(self):
        """Number of feeds in the u-direction."""
        return self.num_cylinders

    @property
    def nfeeds_v(self):
        """Number of feeds in the v-direction."""
        return self.num_feeds



class UnpolarisedCylinderFFTTelescope(CylinderFFTTelescope, UnpolarisedFFTTelescope):
    """A complete class for an Unpolarised Cylinder type Fast Fourier Transform telescope.
    """

    pass



class PolarisedCylinderFFTTelescope(CylinderFFTTelescope, PolarisedFFTTelescope):
    """A complete class for an Polarised Cylinder type Fast Fourier Transform telescope.
    """

    pass
