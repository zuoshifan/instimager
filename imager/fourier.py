import abc
import numpy as np
import healpy as hp
import h5py

from cora.util import coord

from caput import config
from caput import mpiutil

import telescope
import cylinder
import exotic_cylinder
import visibility
import rotate as rot
import fouriertransform as ft



class FourierTransformTelescope(telescope.TransitTelescope):
    """Common functionality for all Fourier Transform Telescopes.

    """

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class

    t_int = config.Property(proptype=float, default=300.0) # integrating time, Unit:s

    @abc.abstractproperty
    def blvector(self):
        """Unique baselines vector of the array including conjugate baselines."""
        return

    @abc.abstractproperty
    def blredundancy(self):
        """The number of each unique baseline corresponds to blvector."""
        return

    @property
    def k(self):
        """The central wavevector magnitude of each frequency band (in metres^-1)."""
        return 2 * np.pi / self.wavelengths

    @abc.abstractproperty
    def Aeff(self):
        """Effective collecting area of each element, Unit: m^2."""
        return

    ################# For generating visibilities ##############

    def _fringe(self, baselines, f_index):
        ## The exponential fringes of some baselines.

        # Get baseline separation and fringe map.
        uv = baselines / self.wavelengths[f_index]
        shp = uv.shape
        uv = uv.reshape(-1, 2).T.reshape((2,) + shp[:-1])
        return visibility.fringe(self._angpos, self.zenith, uv)

    _skymap = None

    @property
    def skymap(self):
        """The sky map to generate simulated visibilities."""
        return self._skymap

    def _read_in_data_from_h5files(self, files, fi_range):
        ## Read in data contained in HDF5 files.
        files = list(files)
        if len(files) == 0:
            raise ValueError('No input files')

        sfreq, efreq = fi_range

        data = None
        for fl in files:
            with h5py.File(fl, 'r') as f:
                if data is None:
                    data = f['map'][sfreq:efreq]
                else:
                    data += f['map'][sfreq:efreq]

        return data

    _original_map = None

    def load_skymap(self, mapfiles, fi_range, nside=64):
        """Load sky map from a list of files for a range of frequencies. If no
        input map files, a zero Healpix map with NSIDE=`nside` will be created."""
        try:
            hpmap = self._read_in_data_from_h5files(mapfiles, fi_range)
        except ValueError:
            warnings.warn('No input sky maps, return a zeros sky map instead')
            # initialize angular positions in healpix map
            self._init_trans(nside)
            self._original_map = np.zeros((len(range(fi_range[0], fi_range[1])), 4, 12*nside**2), dtype=np.float64)

            return

        shp = hpmap.shape
        if shp != 3 and shp[1] != 4:
            raise ValueError('Unsupported sky map file')
        local_nfreq = np.array(shp[0])
        nfreq = np.array(0)
        mpiutil.Allreduce(local_nfreq, nfreq)
        if nfreq != self.nfreq:
            raise ValueError('Input sky map has different frequency channels with the observing frequencies')
        nside = hp.npix2nside(shp[-1]) # to see if an valid number of pixels

        # initialize angular positions in healpix map
        self._init_trans(nside)

        self._original_map = hpmap

    _rotate_angle = 0

    def rotate_skymap(self, angle=0):
        """Rotate the sky map along the longitudinal direction by `angle` degree."""
        self._rotate_angle = angle
        self._skymap = rot.rotate_map(self._original_map, rot=(angle, 0.0, 0.0))

    @abc.abstractmethod
    def pack_skymap(self):
        """Pack the skymap to appropriate format."""
        return

    @abc.abstractmethod
    def gen_visibily_fi(self, fi_range, add_noise=True):
        """Generate simulated visibilities for a range of frequencies for an input sky map."""
        return

    def gen_visibily(self, mapfiles, rot_ang=0, add_noise=True):
        """Generate simulated visibilities for all observing frequencies."""

        nfreqs, sfreqs, efreqs = mpiutil.split_all(self.nfreq)
        nfreq, sfreq, efreq = mpiutil.split_local(self.nfreq)
        lfrange = (sfreq, efreq) # local frequency range

        # Load the input maps
        self.load_skymap(mapfiles, lfrange)
        # rotate the sky map
        self.rotate_skymap(rot_ang)
        # Pack the skymap to appropriate format.
        self.pack_skymap()

        local_vis = self.gen_visibily_fi(lfrange, add_noise=add_noise)
        shp = local_vis.shape
        vis = None
        if mpiutil.rank0:
            vis = np.zeros((self.nfreq,) + shp[1:], dtype=local_vis.dtype)

        sizes = nfreqs * np.prod(shp[1:])
        displ = sfreqs * np.prod(shp[1:])
        dtype = local_vis.dtype
        mpiutil.Gatherv(local_vis, [vis, sizes, displ, mpiutil.typemap(dtype)], root=0)

        return vis

    ################### For map-making ########################

    @abc.abstractmethod
    def map_making_fi(self, vis_range, fi_range, rot_ang=0, divide_beam=True):
        """Map-making for a range of frequencies for the input visibilities."""
        return

    def map_making(self, vis, rot_ang=0, divide_beam=True):
        """Map-making for all observing frequencies."""

        nfreqs, sfreqs, efreqs = mpiutil.split_all(self.nfreq)
        nfreq, sfreq, efreq = mpiutil.split_local(self.nfreq)
        lfrange = (sfreq, efreq) # local frequency range

        local_map = self.map_making_fi(vis[sfreq:efreq], lfrange, rot_ang=rot_ang, divide_beam=divide_beam)
        shp = local_map.shape
        maps = None
        if mpiutil.rank0:
            maps = np.zeros((self.nfreq,) + shp[1:], dtype=local_map.dtype)

        sizes = nfreqs * np.prod(shp[1:])
        displ = sfreqs * np.prod(shp[1:])
        dtype = local_map.dtype
        mpiutil.Gatherv(local_map, [maps, sizes, displ, mpiutil.typemap(dtype)], root=0)

        return maps

    @abc.abstractmethod
    def qvector(self, f_index):
        """The q vector for Fourier transform map-making. vec(q) = (k_x, k_y)."""
        return

    def k_z(self, f_index):
        """The magnitude of k_z for corresponding qvector. k_z = sqrt(k^2 - q^2)."""
        q = self.qvector(f_index)
        q2 = q[..., 0]**2 + q[..., 1]**2

        return np.sqrt(self.k[f_index]**2 - q2)

    def kk_z(self, f_index):
        """Return k * sqrt(k**2 - q**2)."""
        return self.k[f_index] * self.k_z(f_index)

    def hp_pix(self, f_index):
        """The corresponding healpix map pixel for vector k = (k_x, k_y, k_z).
        """
        # unit vectors in equatorial coordinate
        zhat = coord.sph_to_cart(self.zenith)
        uhat, vhat = visibility.uv_plane_cart(self.zenith)

        # convert k-vectors in local coordinate to equatorial coordinate
        q = self.qvector(f_index)
        kz = self.k_z(f_index)
        shp = q.shape
        q = q.reshape(-1, 2).T.reshape(2, shp[:-1])
        k = (np.outer(q[0], uhat) + np.outer(q[1], vhat) + np.outer(kz, zhat)).reshape(kz.shape + zhat.shape)

        return hp.vec2pix(self._nside, k[..., 0], k[..., 1], k[..., 2])

    ###################### Noise related ######################

    def _noise_amp(self, f_index):
        ## Noise temperature amplitude for one frequency channel, Unit: K.
        ## Calculated as lambda^2 * T_sys / A_eff * sqrt(delta_nu * t).
        band_width = self.freq_upper - self.freq_lower # MHz
        return self.wavelengths[f_index]**2 *self.tsys(f_index) / (self.Aeff * np.sqrt(1.0e6 * band_width * self.t_int))

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

    _blvector = None

    @property
    def blvector(self):
        """Unique baselines vector of the array including conjugate baselines."""
        if self._blvector is None:
            bl = self.baselines
            if self.auto_correlations:
                bl = np.concatenate((-bl[:0:-1], bl), axis=0) # avoid repeated [0, 0] baseline
            else:
                bl = np.concatenate((-bl[::-1], bl), axis=0)

            self._blvector = bl

        return self._blvector

    _blredundancy = None

    @property
    def blredundancy(self):
        """The number of each unique baseline corresponds to blvector."""
        if self._blredundancy is None:
            rd = self.redundancy
            if self.auto_correlations:
                rd = np.concatenate((rd[:0:-1], rd)) # take care of [0, 0] baseline
            else:
                rd = np.concatenate((rd[::-1], rd))

            self._blredundancy = rd

        return self._blredundancy

    ################# For generating visibilities ##############

    def pack_skymap(self):
        """Pack the skymap to appropriate format.

        For unpolarised telescope the shape of the packed sky map is (nfreq, npix).
        """
        self._skymap = self._original_map[:, 0, :]


    def gen_visibily_fi(self, fi_range, add_noise=True):
        """Generate simulated visibilities for a range of frequencies for an input sky map."""
        fi_list = range(fi_range[0], fi_range[1])
        nfi = len(fi_list)
        shp = self.beam_map(0).shape
        vis = np.zeros((nfi,) + shp[:-1], dtype=np.complex128)

        for (idx, f_index) in enumerate(fi_list):
            bfi = self.beam_map(f_index)
            vis[idx] = (bfi * self.skymap[idx]).sum(axis=-1) * (4 * np.pi / self.skymap.shape[-1])

            if add_noise:
                vis[idx] += telescope.noise(f_index)

        return vis

    def single_beam(self, f_index):
        """A healpix map of the primary beam."""
        return self.beam(0, f_index) # all feeds are the same

    def beam_solid_angle(self, f_index):
        """Solid angle of the primary beam."""
        beam = self.single_beam(f_index)
        return (np.abs(beam)**2 * self._horizon).sum() * (4*np.pi / beam.size)

    def _beam_map(self, baselines, f_index):
        ## Return a heapix map of |A(n)|^2 * e^(2 * pi * i * n * u_ij) / Omega
        ## for an array of baseline `baselines` in one frequency channel.

        beam = self.single_beam(f_index) # all feeds are the same
        fringe = self._fringe(baselines, f_index)

        # Beam solid angle (integrate over beam^2 - equal area pixels)
        omega_A = self.beam_solid_angle(f_index)

        return self._horizon * fringe * np.abs(beam)**2 / omega_A

    def beam_map(self, f_index):
        """Return a heapix map of |A(n)|^2 * e^(2 * pi * i * n * u_ij) / Omega
        for all baselines in one frequency channel.

        Shape (n_bl, n_pix).

        """
        return self._beam_map(self.blvector, f_index)

    ################### For map-making ########################

    _threshould = 0.0

    def qvector(self, f_index):
        """The q vector for Fourier transform map-making. vec(q) = (k_x, k_y)."""
        beam = self.single_beam(f_index)
        # select index where beam response larger than the given threshould
        (idx,) = np.where(beam >= self._threshould * beam.max())
        nvec = hp.pix2vec(self._nside, idx)

        # unit vectors in equatorial coordinate
        # zhat = coord.sph_to_cart(self.zenith)
        uhat, vhat = visibility.uv_plane_cart(self.zenith)
        qxhat = nvec[0] * uhat[0] + nvec[1] * uhat[1] + nvec[2] * uhat[2]
        qyhat = nvec[0] * vhat[0] + nvec[1] * vhat[1] + nvec[2] * vhat[2]
        q = np.zeros(qxhat.shape + (2,), dtype=qxhat.dtype)
        q[..., 0] = self.k[f_index] * qxhat
        q[..., 1] = self.k[f_index] * qyhat

        return q

    def beam_prod(self, f_index):
        """Return |A(q)|^2 / Omega in the q_grid.

        Shape (nbls_v, nbls_u).

        """
        beam = self.single_beam(f_index) # all feeds are the same
        # Beam solid angle (integrate over beam^2 - equal area pixels)
        omega_A = self.beam_solid_angle(f_index)

        prod = self._horizon * np.abs(beam)**2 / omega_A

        return prod[self.hp_pix(f_index)]

    def map_making_fi(self, vis_range, fi_range, rot_ang=0, divide_beam=True):
        """Map-making for a range of frequencies for the input visibilities."""

        fi_list = range(fi_range[0], fi_range[1])
        nfi = len(fi_list)
        T_map = np.zeros((nfi, 4, 12 * self._nside**2), dtype=np.float64)

        for (idx, f_index) in enumerate(fi_list):
            qvector = self.qvector(f_index)
            # ft_vis = np.zeros(qvector.shape[0], dtype=np.complex128)
            vis_fi = vis_range[idx]
            # for (qi, q) in enumerate(qvector):
            #     for (bi, bl) in enumerate(self.blvector):
            #         rd = self.blredundancy[bi] # baseline redundancy
            #         ft_vis[qi] += rd * vis_fi[bi] * np.exp(-1.0J * (q[0] * bl[0] + q[1] * bl[1]))

            # ft_vis /= np.sum(self.blredundancy)
            # ft_vis = ft_vis.real # only the real part

            dirty_T = ft.ft_vis(vis_fi, qvector, self.blvector, self.blredundancy)

            kk_z = self.kk_z(f_index)
            T = kk_z * dirty_T  # actually (|A|^2 / Omega) * T (dirty map)
            if divide_beam:
                beam_prod = self.beam_prod(f_index)
                T = np.ma.divide(T, beam_prod) # clean map
                # T /= beam_prod

            T_map[idx, 0, self.hp_pix(f_index)] = T # only T

            # inversely rotate the sky map
            T_map = rot.rotate_map(T_map, rot=(rot_ang, 0.0, 0.0))

        return np.ascontiguousarray(T_map) # Return a contiguous array in memory (C order) as mpi4py requires that


    ###################### Noise related ######################

    def _noise(self, baselines, f_index):
        ## Noise temperature for some baselines for one frequency channel, Unit: K
        shp = baseline.shape[:-1]
        cnormal = np.random.normal(size=shp) + 1.0J * np.random.normal(size=shp)
        return self._noise_amp(f_index) * cnormal # maybe divide sqrt(2)

    def noise(self, f_index):
        """Noise temperature for all baselines for one frequency channel, Unit: K
        """
        return self._noise(self.blvector, f_index)




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




class UnpolarisedCylinderFourierTransformTelescope(exotic_cylinder.ArbitraryUnpolarisedCylinder, UnpolarisedFourierTransformTelescope):
    """A complete class for an Unpolarised Cylinder type Fourier Transform telescope.
    """

    def Aeff(self):
        """Effective collecting area of each element, Unit: m^2."""
        average_fd_spacing = np.dot(self.num_feeds, self.feed_spacing) / np.sum(self.num_feeds)

        return self.cylinder_width * average_fd_spacing




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

    @property
    def blvector(self):
        return self._bl_grid

    def qvector(self, f_index):
        return self._q_grid

    @property
    def _bl_grid(self):
        ## The baseline grid.

        ## Packed as array([[[u1, v1], [u2, v1], [u3, v1], ...],
        ##                  [[u1, v2], [u2, v2], [u3, v2], ...],
        ##                  [[u1, v3], [u2, v3], [u3, v3], ...],
        ##                  ...])
        ## with shape (nbls_v, nbls_u, 2).

        bl = [ np.array([iu * self.delta_u, iv * self.delta_v]) for iv in range(-(self.nfeeds_v - 1), self.nfeeds_v) for iu in range(-(self.nfeeds_u - 1), self.nfeeds_u) ]
        bl = np.array(bl).reshape((self.nbls_v, self.nbls_u, 2))

        return bl

    @property
    def _q_grid(self):
        ## The q vector grid (k_x, k_y).

        ## Packed as array([[[k_x1, k_y1], [k_x2, k_y1], ...],
        ##                  [[k_x1, k_y2], [k_x2, k_y2], ...],
        ##                  ...])
        ## with shape (nbls_v, nbls_u, 2).

        delta_kx = 2 * np.pi / (self.nbls_u * self.delta_u)
        delta_ky = 2 * np.pi / (self.nbls_v * self.delta_v)

        q = [ np.array([ix * delta_kx, iy * delta_ky]) for iy in range(-(self.nfeeds_v - 1), self.nfeeds_v) for ix in range(-(self.nfeeds_u - 1), self.nfeeds_u) ]
        q = np.array(q).reshape((self.nbls_v, self.nbls_u, 2))

        return q




class UnpolarisedFFTTelescope(FFTTelescope, UnpolarisedFourierTransformTelescope):
    """A base for a unpolarised Fast Fourier transform telescope.

    """

    pass




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

    def map_making_fi(self, vis_fi, f_index, rot_ang=0, divide_beam=True):
        """Map-making for one frequency for the input visibilities."""

        # first arrange data according to fftfreq
        vis_fi = np.fft.ifftshift(vis_fi)
        fft_vis = np.prod(vis_fi.shape) * np.fft.ifft2(vis_fi).real
        # fft_vis = np.fft.ifftshift(fft_vis)
        # fft_vis = np.fft.fftshift(fft_vis)

        kk_z = self.kk_z(f_index)
        T_grid = kk_z * fft_vis # actually (|A|^2 / Omega) * T (dirty map)
        if divide_beam:
            beam_prod = self.beam_prod(f_index)
            T_grid = np.ma.divide(T_grid, beam_prod) # clean map

        # convert to healpix map
        T_map = np.zeros((4, 12 * self._nside**2), dtype=T_grid.dtype)
        T_map[0, self.hp_pix(f_index)] = T_grid # only T

        # inversely rotate the sky map
        T_map = rot.rotate_map(T_map, rot=(rot_ang, 0.0, 0.0))

        return T_map




class PolarisedCylinderFFTTelescope(CylinderFFTTelescope, cylinder.PolarisedCylinderTelescope, PolarisedFFTTelescope):
    """A complete class for an Polarised Cylinder type Fast Fourier Transform telescope.
    """

    pass
