import abc
import numpy as np
import healpy as hp

import telescope
import cylinder
import visibility

from cora.util import coord

from caput import config



class FourierTransformTelescope(telescope.TransitTelescope):
    """Common functionality for all Fourier Transform Telescopes.

    """

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class

    t_int = config.Property(proptype=float, default=300.0) # integrating time, Unit:s


    @property
    def k(self):
        """The central wavevector magnitude of each frequency band (in metres^-1)."""
        return 2 * np.pi / self.wavelengths

    # @abc.abstractproperty
    # def t(self):
    #     """Integrating time, Unit: s."""
    #     return

    @abc.abstractproperty
    def Aeff(self):
        """Effective collecting area of each element, Unit: m^2."""
        return

    def noise_amp(self, f_index):
        """Noise temperature amplitude for one frequency channel, Unit: K.

        Calculated as lambda^2 * T_sys / A_eff * sqrt(delta_nu * t).
        """
        band_width = self.freq_upper - self.freq_lower # MHz
        return self.wavelengths[f_index]**2 *self.tsys(f_index) / (self.Aeff * np.sqrt(1.0e6 * band_width * self.t_int))

    def _noise(self, baselines, f_index):
        ## Noise temperature for an array of baselines for one frequency channel, Unit: K

        # complex noise, maybe divide by sqrt(2) ???
        return self.noise_amp(f_index) * (np.random.normal(size=baselines.shape[:-1]) + 1.0J * np.random.normal(size=baselines.shape[:-1]))





class UnpolarisedFourierTransformTelescope(FourierTransformTelescope, telescope.SimpleUnpolarisedTelescope):
    """A base for a unpolarised Fourier transform telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beam`.

    Abstract Methods
    ----------------
    beam : method
        Routines giving the field pattern for the feeds.
    """

    def _beam_map(self, baselines, f_index):
        """Return a heapix map of |A(n)|^2 * e^(2 * pi * i * n * u_ij) / Omega
        for an array of baseline `baselines` in one frequency channel.

        """
        beam = self.beam(0, f_index) # all feeds are the same
        # Get baseline separation and fringe map.
        uv = baselines / self.wavelengths[f_index]
        shp = uv.shape
        uv = uv.reshape(-1, 2).T.reshape((2,) + shp[:-1])
        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        import h5py
        with h5py.File('fringe_%d.hdf5' % f_index, 'w') as f:
            f.create_dataset('map', data=fringe[0, 0])

        # Beam solid angle (integrate over beam^2 - equal area pixels)
        omega_A = (np.abs(beam)**2 * self._horizon).sum() * (4*np.pi / beam.size)

        return self._horizon * fringe * np.abs(beam)**2 / omega_A

    def beam_map(self, f_index):
        """Return a heapix map of |A(n)|^2 * e^(2 * pi * i * n * u_ij) / Omega
        for all baselines in one frequency channel.

        Shape (n_bl, n_pix).

        """
        return self._beam_map(self.baselines, f_index)

    def noise(self, f_index):
        """Noise temperature for all baselines for one frequency channel, Unit: K
        """
        return self._noise(self.baselines, f_index)





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
    def nbls_u(self):
        """Number of points in the baseline grid in the u-directions."""
        return 2 * self.nfeeds_u - 1

    @property
    def nbls_v(self):
        """Number of points in the baseline grid in the v-directions."""
        return 2 * self.nfeeds_v - 1

    @property
    def bl_grid(self):
        """The baseline grid.

        Packed as array([[[u1, v1], [u2, v1], [u3, v1], ...],
                         [[u1, v2], [u2, v2], [u3, v2], ...],
                         [[u1, v3], [u2, v3], [u3, v3], ...],
                         ...])
        with shape (nbls_v, nbls_u, 2).

        """
        bl = [ np.array([iu * self.delta_u, iv * self.delta_v]) for iv in range(-(self.nfeeds_v - 1), self.nfeeds_v) for iu in range(-(self.nfeeds_u - 1), self.nfeeds_u) ]
        bl = np.array(bl).reshape((self.nbls_v, self.nbls_u, 2))

        return bl

    @property
    def q_grid(self):
        """The q vector grid (k_x, k_y).

        Packed as array([[[k_x1, k_y1], [k_x2, k_y1], ...],
                         [[k_x1, k_y2], [k_x2, k_y2], ...],
                         ...])
        with shape (nbls_v, nbls_u, 2).

        """
        delta_kx = 2 * np.pi / (self.nbls_u * self.delta_u)
        delta_ky = 2 * np.pi / (self.nbls_v * self.delta_v)

        q = [ np.array([ix * delta_kx, iy * delta_ky]) for iy in range(-(self.nfeeds_v - 1), self.nfeeds_v) for ix in range(-(self.nfeeds_u - 1), self.nfeeds_u) ]
        q = np.array(q).reshape((self.nbls_v, self.nbls_u, 2))

        return q

    def k_z(self, ifreq):
        """The magnitude of k_z for frequency channel `ifreq`. k_z = sqrt(k^2 - q^2).

        Shape (nbls_v, nbls_u).

        """
        q2 = self.q_grid[:, :, 0]**2 + self.q_grid[:, :, 1]**2

        return np.sqrt(self.k[ifreq]**2 - q2)

    def hp_pix(self, ifreq):
        """The corresponding healpix map pixel for vector k = (k_x, k_y, kz).

        Shape (nbls_v, nbls_u).

        """
        # unit vectors in equatorial coordinate
        zhat = coord.sph_to_cart(self.zenith)
        uhat, vhat = visibility.uv_plane_cart(self.zenith)

        # convert k-vectors in local coordinate to equatorial coordinate
        kx = self.q_grid[:, :, 0] * uhat[0] + self.q_grid[:, :, 1] * uhat[1] + self.k_z(ifreq) * uhat[2]
        ky = self.q_grid[:, :, 0] * vhat[0] + self.q_grid[:, :, 1] * vhat[1] + self.k_z(ifreq) * vhat[2]
        kz = self.q_grid[:, :, 0] * zhat[0] + self.q_grid[:, :, 1] * zhat[1] + self.k_z(ifreq) * zhat[2]

        # return hp.vec2pix(self._nside, kx, ky, kz)
        # note the direction of -vec(k) is vec(n)
        return hp.vec2pix(self._nside, -kx, -ky, kz)

        # lat, lon = self.zenith
        # lat = np.pi / 2 - lat
        # kx = self.q_grid[:, :, 0]
        # ky = self.q_grid[:, :, 1]
        # kz = self.k_z(ifreq)
        # eqkx = -ky * np.sin(lat) + kz * np.cos(lat)
        # eqky = ky
        # eqkz = ky * np.cos(lat) + kz * np.sin(lat)

        # eqkx, eqky, eqkz = hp.Rotator(rot=(-lon, 0.0, 0.0))(eqkx, eqky, eqkz)

        # return hp.vec2pix(self._nside, eqkx, eqky, eqkz)

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

    def beam_map(self, f_index):
        """Return a heapix map of |A(n)|^2 * e^(2 * pi * i * n * u_ij) / Omega
        for all baselines in one frequency channel.

        Shape (nbls_v, nbls_u, n_pix).

        """
        return self._beam_map(self.bl_grid, f_index)

    def beam_prod(self, ifreq):
        """Return |A(q)|^2 / (Omega * k * sqrt(k^2 - q^2)) in the q_grid.

        Shape (nbls_v, nbls_u).

        """
        beam = self.beam(0, ifreq) # all feeds are the same
        # Beam solid angle (integrate over beam^2 - equal area pixels)
        omega_A = (np.abs(beam)**2 * self._horizon).sum() * (4*np.pi / beam.size)

        prod = self._horizon * np.abs(beam)**2 / omega_A
        kk_z = self.k[ifreq] * self.k_z(ifreq)

        return prod[self.hp_pix(ifreq)] / kk_z

    def noise(self, f_index):
        """Noise temperature for all baselines for one frequency channel, Unit: K
        """
        return self._noise(self.bl_grid, f_index)








class PolarisedFFTTelescope(FFTTelescope, PolarisedFourierTransformTelescope):
    """A base for a polarised Fast Fourier transform telescope.

    """

    pass



class CylinderFFTTelescope(FFTTelescope, cylinder.CylinderTelescope):
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

    @property
    def Aeff(self):
        """Effective collecting area of each feed, Unit: m^2."""
        return self.cylinder_width * self.delta_v




class UnpolarisedCylinderFFTTelescope(CylinderFFTTelescope, cylinder.UnpolarisedCylinderTelescope, UnpolarisedFFTTelescope):
    """A complete class for an Unpolarised Cylinder type Fast Fourier Transform telescope.
    """

    ## u-width property override
    @property
    def u_width(self):
        return self.cylinder_width

    ## v-width property override
    @property
    def v_width(self):
        return 0.0




class PolarisedCylinderFFTTelescope(CylinderFFTTelescope, cylinder.PolarisedCylinderTelescope, PolarisedFFTTelescope):
    """A complete class for an Polarised Cylinder type Fast Fourier Transform telescope.
    """

    pass
