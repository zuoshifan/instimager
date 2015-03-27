import abc
import warnings

import numpy as np
import healpy as hp
import h5py


def real_equiv(dtype):
    ## Return the real datatype with the same precision as dtype.
    if dtype == np.float32 or dtype == np.complex64:
        return np.float32

    if dtype == np.float64 or dtype == np.complex128:
        return np.float64

    raise ValueError("Unsupported data type")


def complex_equiv(dtype):
    ## Return the complex datatype with the same precision as dtype.
    if dtype == np.float32 or dtype == np.complex64:
        return np.complex64

    if dtype == np.float64 or dtype == np.complex128:
        return np.complex128

    raise ValueError("Unsupported data type")



class BrightnessMap(object):
    """Base class for a sky brightness map."""

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class

    _nside = None # Healpix NSIDE
    _nfreq = None # number of frequencies
    _dtype = None
    _skymap = None

    def __init__(self, nfreq, nside=64, dtype=np.float64):
        self._nfreq = nfreq
        self._nside = nside
        self._dtype = self.type_convert(dtype)
        self._skymap =  np.zeros(self.shape, dtype=self.dtype)

    @abc.abstractmethod
    @staticmethod
    def type_convert(dtype):
        return

    @property
    def pol(self):
        return self._pol

    @property
    def nside(self):
        return self._nside

    @property
    def npix(self):
        return 12 * self.nside**2

    @property
    def nfreq(self):
        return self._nfreq

    @abc.abstractproperty
    def shape(self):
        return

    @property
    def dtype(self):
        return self._dtype

    @abc.property
    def sky_map(self):
        return self._skymap

    @classmethod
    def copy(cls):
        new_map = cls(self.nfreq, self.nside, self.dtype)
        new_map._skymap = self._skymap.copy()

        return new_map

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

    @abc.abstractmethod
    @classmethod
    def read_from_h5files(cls, files):
        """Read from a list of HDF5 files containing the sky maps."""
        return

    @abc.abstractmethod
    def write_to_h5file(self, filename):
        """Write the sky map to a HDF5 file."""
        return


class UnpolarisedBrightnessMap(BrightnessMap):
    """Class for an unpolarised sky brightness map."""

    _pol = False

    @staticmethod
    def type_convert(dtype):
        return real_equiv(dtype)

    @abc.property
    def shape(self):
        return (self.nfreq, 1, 1, self.npix)

    @classmethod
    def read_from_h5files(cls, files):
        """Read from a list of HDF5 files containing the sky maps."""
        try:
            hpmap = self._read_in_data_from_h5files(files)
        except ValueError:
            warnings.warn('No input sky maps, return a zeros sky map instead')
            return cls(self.nfreq, self.nside, self.dtype)

        shp = hpmap.shape
        if shp != 3:
            raise ValueError('Unsupported sky map file')
        nfreq = shp[0]
        nside = hp.npix2nside(shp[-1])
        dtype = hpmap.dtype

        self = cls(nfreq, nside, dtype)
        self._skymap += hpmap[:, 0, :] # T

        return self

    def write_to_h5file(self, filename):
        """Write the sky map to a HDF5 file."""
        hpmap = np.zeros((self.nfreq, 4, self.npix), dtype=self.dtype)
        hpmap[:, 0, :] = self._skymap
        with h5py.File(filename, 'w') as f:
            f.create_dataset('map', data=hpmap)

    def hconj(self):
        """Hermitian conjugate of the sky map."""
        return self.copy()

    def inv(self):
        """Inverse of the sky map."""
        new_map = self.copy()
        new_map._skymap = 1.0 / self._skymap

        return new_map


class PolarisedBrightnessMap(BrightnessMap):
    """Class for an unpolarised sky brightness map."""

    _pol = True

    @staticmethod
    def type_convert(dtype):
        return complex_equiv(dtype)

    @abc.property
    def shape(self):
        return (self.nfreq, 2, 2, self.npix)

    @classmethod
    def read_from_h5files(cls, files):
        """Read from a list of HDF5 files containing the sky maps."""
        try:
            hpmap = self._read_in_data_from_h5files(files)
        except ValueError:
            warnings.warn('No input sky maps, return a zeros sky map instead')
            return cls(self.nfreq, self.nside, self.dtype)

        shp = hpmap.shape
        if shp != 3:
            raise ValueError('Unsupported sky map file')
        nfreq = shp[0]
        nside = hp.npix2nside(shp[-1])
        dtype = hpmap.dtype

        self = cls(nfreq, nside, dtype)
        # adopt the opposite convention
        self._skymap[:, 0, 0, :] += (hpmap[:, 0, :] + hpmap[:, 1, :]) # T + Q
        self._skymap[:, 0, 1, :] += (hpmap[:, 2, :] - 1.0J * hpmap[:, 3, :]) # U - iV
        self._skymap[:, 1, 0, :] += (hpmap[:, 2, :] + 1.0J * hpmap[:, 3, :]) # U + iV
        self._skymap[:, 1, 1, :] += (hpmap[:, 0, :] - hpmap[:, 1, :]) # T - Q

        return self

    def write_to_h5file(self, filename):
        """Write the sky map to a HDF5 file."""
        hpmap = np.zeros((self.nfreq, 4, self.npix), dtype=real_equiv(self.dtype))
        hpmap[:, 0, :] = (self._skymap[:, 0, 0, :] + self._skymap[:, 1, 1, :]).real / 2.0 # T
        hpmap[:, 1, :] = (self._skymap[:, 0, 0, :] - self._skymap[:, 1, 1, :]).real / 2.0 # Q
        hpmap[:, 2, :] = (self._skymap[:, 0, 1, :] + self._skymap[:, 1, 0, :]).real / 2.0 # U
        hpmap[:, 3, :] = - (self._skymap[:, 0, 1, :] - self._skymap[:, 1, 0, :]).imag / 2.0 # V

        with h5py.File(filename, 'w') as f:
            f.create_dataset('map', data=hpmap)

    def hconj(self):
        """Hermitian conjugate of the sky map."""
        new_map = self.copy()
        new_map._skymap[:, 0, 0, :] = self._skymap[:, 0, 0, :].conj()
        new_map._skymap[:, 0, 1, :] = self._skymap[:, 1, 0, :].conj()
        new_map._skymap[:, 1, 0, :] = self._skymap[:, 0, 1, :].conj()
        new_map._skymap[:, 1, 1, :] = self._skymap[:, 1, 1, :].conj()
        return new_map

    def inv(self):
        """Inverse of the sky map."""
        new_map = self.copy()
        for fi in range(self.nfreq):
            for pix in range(self.npix):
                new_map._skymap[fi, :, :, pix] = np.inv(self._skymap[fi, :, :, pix])

        return new_map
