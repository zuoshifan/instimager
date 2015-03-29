import abc
import numpy as np
import healpy as hp

from cora.util import coord

from caput import config

import telescope
import cylinder
import visibility
import rotate as rot



class FourierTransformTelescope(telescope.TransitTelescope):
    """Common functionality for all Fourier Transform Telescopes.

    """

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class

    t_int = config.Property(proptype=float, default=300.0) # integrating time, Unit:s


    @property
    def k(self):
        """The central wavevector magnitude of each frequency band (in metres^-1)."""
        return 2 * np.pi / self.wavelengths

    @abc.abstractproperty
    def Aeff(self):
        """Effective collecting area of each element, Unit: m^2."""
        return

    # @abc.abstractmethod
    # def single_beam(self, f_index):
    #     """Primary beam response of the telescope.

    #     A healpix mapfor an unpolarised telescope, a (2, 2) healpix map matrix
    #     for a polarised telescope."""
    #     return

    # @abc.abstractmethod
    # def beam_solid_angle(self, f_index):
    #     """Solid angle of the primary beam."""
    #     beam = self.single_beam(f_index)
    #     return (np.abs(beam)**2 * self._horizon).sum() * (4*np.pi / beam.size)

    def fringe(self, baselines, f_index):
        """The exponential fringes of some baselines."""

        # Get baseline separation and fringe map.
        uv = baselines / self.wavelengths[f_index]
        shp = uv.shape
        uv = uv.reshape(-1, 2).T.reshape((2,) + shp[:-1])
        return visibility.fringe(self._angpos, self.zenith, uv)

    # @abc.abstractmethod
    # @staticmethod
    # def prod(x1, x2):
    #     """Return the product of `x1` and `x2`."""
    #     return x1 * x2

    # @abc.abstractmethod
    # @staticmethod
    # def inv(x):
    #     """Return the inverse of `x`."""
    #     return 1.0 / x

    # @abc.abstractmethod
    # @staticmethod
    # def hconj(x):
    #     """Return the Hermitian conjugate of `x`."""
    #     return np.conj(x)

    # def vis(self, sky_map, baselines, f_index, add_noise=True):
    #     """The observed visibilities of some baselines given a sky map `sky_map`."""
    #     beam = self.single_beam(f_index)
    #     BSBdagger = self.prod(self.prod(beam, sky_map), self.hconj(beam))
    #     fringe = self.fringe(baselines, f_index)
    #     beam_ang = self.beam_solid_angle(f_index)

    #     vis = (BSBdagger * fringe / beam_ang).sum(axis=-1) * (4 * pi / sky_map.shape[-1])
    #     if add_noise:
    #         vis += self._noise(baselines, f_index)

    #     return vis

    @abc.abstractproperty
    def blvector(self):
        """Baselines vector of the array."""
        return self.baselines

    _skymap = None

    @property
    def skymap(self):
        """The sky map to generate simulated visibilities."""
        return self._skymap

    # @abc.abstractmethod
    # def load_skymap(self, mapfiles=[], nside=64):
    #     """Load sky map from a list of files. If no input map files, a zero
    #     Healpix map with NSIDE=`nside` will be generated."""
    #     return

    @staticmethod
    def _read_in_data_from_h5files(files):
        ## Read in data contained in HDF5 files.
        files = list(files)
        if len(file) == 0:
            raise ValueError('No input files')
        # read meta data from the first file
        data = None
        for fl in files:
            with h5py.File(files, 'r') as f:
                if data is None:
                    data = f['map'][...]
                else:
                    data += f['map'][...]

        return data

    _original_map = None

    def load_skymap(self, mapfiles, nside=64):
        """Load sky map from a list of files. If no input map files, a zero
        Healpix map with NSIDE=`nside` will be generated."""
        try:
            hpmap = self._read_in_data_from_h5files(files)
        except ValueError:
            warnings.warn('No input sky maps, return a zeros sky map instead')
            # initialize angular positions in healpix map
            self._init_trans(nside)
            self._original_map = np.zeros((self.nfreq, 4, 12*nside**2), dtype=np.float64)

        shp = hpmap.shape
        if shp != 3 and shp[1] != 4:
            raise ValueError('Unsupported sky map file')
        nfreq = shp[0]
        if nfreq != self.nfreq:
            raise ValueError('Input sky map has different frequency channels with the observing frequencies')
        nside = hp.npix2nside(shp[-1]) # to see if an valid number of pixels

        # initialize angular positions in healpix map
        self._init_trans(nside)

        self._original_map = hpmap


    _rotate_angle = 0

    def rotate_skymap(angle=0):
        """Rotate the sky map along the longitudial direction by `angle` degree."""
        self._rotate_angle = angle
        return rot.rotate_map(self._original_map, rot=(angle, 0.0, 0.0))

    @abc.abstractmethod
    def pack_skymap(self):
        """Pack the skymap to appropriate format."""
        return

    @abc.abstractmethod
    def gen_visibily(self, fi_index, add_noise=True):
        """Generate simulated visibilities at one frequency for an input sky map."""
        return

    @abc.abstractmethod
    def map_making(self, vis_file):
        """Map-making for the input visibilities."""
        return

    @abc.abstractmethod
    def qvector(self, f_index):
        """The q vector for Fourier transform map-making. vec(q) = (k_x, k_y)."""
        return

    def k_z(self, f_index):
        """The magnitude of k_z for corresponding qvector. k_z = sqrt(k^2 - q^2)."""
        q = self.qvector(f_index)
        q2 = q[0]**2 + q[1]**2

        return np.sqrt(self.k(f_index)**2 - q2)

    def hp_pix(self, ifreq):
        """The corresponding healpix map pixel for vector k = (k_x, k_y, kz).
        """
        # unit vectors in equatorial coordinate
        zhat = coord.sph_to_cart(self.zenith)
        uhat, vhat = visibility.uv_plane_cart(self.zenith)

        # convert k-vectors in local coordinate to equatorial coordinate
        q = self.qvector(f_index)
        kz = self.k_z(ifreq)
        shp = q.shape
        q = q.reshape(-1, 2).T.reshape(2, shp[:-1])
        k = (np.outer(q[0], uhat) + np.outer(q[1], vhat) + np.outer(kz, zhat)).reshape(kz.shape + zhat.shape)

        return hp.vec2pix(self._nside, k[..., 0], k[..., 1], k[..., 2])

    # @abc.abstractmethod
    # @staticmethod
    # def fourier_transform(vis):
    #     """Fourier transform `vis` at an array of q-vectors."""
    #     return

    # def map_making(self, vis, f_index):
    #     """Map-making via Fourier transforming the visibilities."""
    #     SBq = self.fourier_transform(vis)
    #     kkz = self.k(f_index) * self.k_z(f_index)
    #     invB = self.inv(self.single_beam(f_index))

    #     Sq = kkz * self.prod(self.prod(self.hconj(invB), SBq), invB)


    def noise_amp(self, f_index):
        """Noise temperature amplitude for one frequency channel, Unit: K.

        Calculated as lambda^2 * T_sys / A_eff * sqrt(delta_nu * t).
        """
        band_width = self.freq_upper - self.freq_lower # MHz
        return self.wavelengths[f_index]**2 *self.tsys(f_index) / (self.Aeff * np.sqrt(1.0e6 * band_width * self.t_int))

    # def _noise(self, baselines, f_index):
    #     ## Noise temperature for an array of baselines for one frequency channel, Unit: K

    #     # complex noise, maybe divide by sqrt(2) ???
    #     return self.noise_amp(f_index) * (np.random.normal(size=baselines.shape[:-1]) + 1.0J * np.random.normal(size=baselines.shape[:-1]))

    @abc.abstractmethod
    def _noise(self, baselines, f_index):
        ## Noise temperature for some baselines for one frequency channel, Unit: K
        return

    @abc.abstractmethod
    def noise(self, f_index):
        """Noise temperature for one frequency channel, Unit: K."""
        return





class UnpolarisedFourierTransformTelescope(FourierTransformTelescope, telescope.SimpleUnpolarisedTelescope):
    """A base for a unpolarised Fourier transform telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beam`.

    Abstract Methods
    ----------------
    beam : method
        Routines giving the field pattern for the feeds.
    """

    def blvector(self):
        """Baselines vector of the array."""
        return self.baselines

    def single_beam(self, f_index):
        """A healpix map of the primary beam."""
        return self.beam(0, f_index) # all feeds are the same

    def beam_solid_angle(self, f_index):
        """Solid angle of the primary beam."""
        beam = self.single_beam(f_index)
        return (np.abs(beam)**2 * self._horizon).sum() * (4*np.pi / beam.size)

    def _beam_map(self, baselines, f_index):
        """Return a heapix map of |A(n)|^2 * e^(2 * pi * i * n * u_ij) / Omega
        for an array of baseline `baselines` in one frequency channel.

        """
        beam = self.single_beam(f_index) # all feeds are the same
        # Get baseline separation and fringe map.
        uv = baselines / self.wavelengths[f_index]
        shp = uv.shape
        uv = uv.reshape(-1, 2).T.reshape((2,) + shp[:-1])
        fringe = self.fringe(uv, f_index)

        # Beam solid angle (integrate over beam^2 - equal area pixels)
        omega_A = self.beam_solid_angle(f_index)

        return self._horizon * fringe * np.abs(beam)**2 / omega_A

    def beam_map(self, f_index):
        """Return a heapix map of |A(n)|^2 * e^(2 * pi * i * n * u_ij) / Omega
        for all baselines in one frequency channel.

        Shape (n_bl, n_pix).

        """
        return self._beam_map(self.blvector, f_index)

    def beam_prod(self, f_index):
        """Return |A(q)|^2 / (Omega * k * sqrt(k^2 - q^2)) in the q_grid.

        Shape (nbls_v, nbls_u).

        """
        beam = self.single_beam(f_index) # all feeds are the same
        # Beam solid angle (integrate over beam^2 - equal area pixels)
        omega_A = self.beam_solid_angle(f_index)

        prod = self._horizon * np.abs(beam)**2 / omega_A
        kk_z = self.k[f_index] * self.k_z(f_index)

        return prod[self.hp_pix(f_index)] / kk_z

    def _noise(self, baselines, f_index):
        ## Noise temperature for some baselines for one frequency channel, Unit: K
        shp = baseline.shape[:-1]
        cnormal = np.random.normal(size=shp) + 1.0J * np.random.normal(size=shp)
        return self.noise_amp(f_index) * cnormal

    def noise(self, f_index):
        """Noise temperature for all baselines for one frequency channel, Unit: K
        """
        return self._noise(self.baseline, f_index)

    def pack_skymap(self):
        """Pack the skymap to appropriate format.

        For unpolarised telescope the shape of the packed sky map is (nfreq, npix).
        """
        self._skymap = self._original_map[:, 0, :]


    def gen_visibily(self, fi_index, add_noise=True):
        """Generate simulated visibilities at one frequency for an input sky map."""
        bfi = self.beam_map(fi_index)
        vis = (bfi * self.sky_map[fi_index]).sum(axis=-1) * (4 * np.pi / hpmap.shape[-1])

        if add_noise:
            vis += telescope.noise(fi)

        return vis




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

    # def k_z(self, ifreq):
    #     """The magnitude of k_z for frequency channel `ifreq`. k_z = sqrt(k^2 - q^2).

    #     Shape (nbls_v, nbls_u).

    #     """
    #     q2 = self.q_grid[:, :, 0]**2 + self.q_grid[:, :, 1]**2

    #     return np.sqrt(self.k[ifreq]**2 - q2)

    # def hp_pix(self, ifreq):
    #     """The corresponding healpix map pixel for vector k = (k_x, k_y, kz).

    #     Shape (nbls_v, nbls_u).

    #     """
    #     # unit vectors in equatorial coordinate
    #     zhat = coord.sph_to_cart(self.zenith)
    #     uhat, vhat = visibility.uv_plane_cart(self.zenith)

    #     # convert k-vectors in local coordinate to equatorial coordinate
    #     kz = self.k_z(ifreq)
    #     shp = self.q_grid.shape
    #     q = self.q_grid.reshape(-1, 2).T.reshape(2, shp[:-1])
    #     k = (np.outer(q[0], uhat) + np.outer(q[1], vhat) + np.outer(kz, zhat)).reshape(kz.shape + zhat.shape)

    #     return hp.vec2pix(self._nside, k[..., 0], k[..., 1], k[..., 2])

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

    # def beam_map(self, f_index):
    #     """Return a heapix map of |A(n)|^2 * e^(2 * pi * i * n * u_ij) / Omega
    #     for all baselines in one frequency channel.

    #     Shape (nbls_v, nbls_u, n_pix).

    #     """
    #     return self._beam_map(self.bl_grid, f_index)

    # def beam_prod(self, ifreq):
    #     """Return |A(q)|^2 / (Omega * k * sqrt(k^2 - q^2)) in the q_grid.

    #     Shape (nbls_v, nbls_u).

    #     """
    #     beam = self.beam(0, ifreq) # all feeds are the same
    #     # Beam solid angle (integrate over beam^2 - equal area pixels)
    #     omega_A = (np.abs(beam)**2 * self._horizon).sum() * (4*np.pi / beam.size)

    #     prod = self._horizon * np.abs(beam)**2 / omega_A
    #     kk_z = self.k[ifreq] * self.k_z(ifreq)

    #     return prod[self.hp_pix(ifreq)] / kk_z

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

    def map_making(self, vis_file):
        """Map-making for the input visibilities."""
        with h5py.File(vis_file, 'r') as f:
            vis = f['vis'][...]
            fi = f.attrs['f_index']

        # first arrange data according to fftfreq
        fft_vis = np.fft.ifftshift(vis)
        fft_vis = np.prod(vis.shape) * np.fft.ifft2(vis).real
        # fft_vis = np.fft.ifftshift(fft_vis)
        # fft_vis = np.fft.fftshift(fft_vis)

        beam_prod = telescope.beam_prod(fi)
        # T_grid = fft_vis / beam_prod
        T_grid = np.ma.divide(fft_vis, beam_prod)

        # convert to healpix map
        T_map = np.zeros((telescope.num_pol_sky, 12 * telescope._nside**2), dtype=T_grid.dtype)
        # T_map[..., telescope.hp_pix(fi)] = T_grid.flatten()
        T_map[..., telescope.hp_pix(fi)] = T_grid

        # inversely rotate the sky map
        T_map = rot.rotate_map(T_map, rot=(telescope.rot_ang, 0.0, 0.0))

        return T_map







class PolarisedCylinderFFTTelescope(CylinderFFTTelescope, cylinder.PolarisedCylinderTelescope, PolarisedFFTTelescope):
    """A complete class for an Polarised Cylinder type Fast Fourier Transform telescope.
    """

    pass
