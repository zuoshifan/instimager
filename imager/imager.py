import numpy as np
import healpy as hp
import h5py

from caput import config
from caput import mpiutil
from caput.pipeline import TaskBase
from caput.pipeline import SingleH5Base
from caput.pipeline import PipelineStopIteration

import fourier
import rotate as rot


sidereal_day = 23.9344696 * 60 * 60 # Unit: s


class InitUnpolCylinderFFTTelescope(fourier.UnpolarisedCylinderFFTTelescope, TaskBase):
    """Initialize an unpolarised cylinder type FFT telescope by reading parameters
    from a YAML configuration file.

    """

    def setup(self):
        self.i = 0
        print 'Initialize an unpolarised cylinder type FFT telescope.'

    def next(self):
        if self.i > 0:
            raise PipelineStopIteration()
        self.i += 1
        return self # return a initialized instance

    def finish(self):
        print 'Initialization done.'



# class GenerateVisibility(SingleH5Base):
class GenerateVisibility(TaskBase):
    """Generate simulated visibilities."""

    maps = config.Property(proptype=list, default=[])
    t_obs = config.Property(proptype=float, default=0.0) # Unit: s
    add_noise = config.Property(proptype=bool, default=True)

    def setup(self):
        print 'Begin to generate simulated visibilities.'

    def next(self, telescope):
        nfreq = telescope.nfreq
        npol = telescope.num_pol_sky

        # the earth rotation angle, equivalently the sky rotates a negative rot_ang
        rot_ang = 360.0 * self.t_obs / sidereal_day # degree

        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
        local_freq = range(sfreq, efreq)

        # read in sky maps
        if len(self.maps) > 0:

            # Load file to find out the map shapes.
            with h5py.File(self.maps[0], 'r') as f:
                mapshape = f['map'].shape

            if lfreq > 0:

                # Allocate array to store the local frequencies
                hpmap = np.zeros((lfreq, npol) + mapshape[2:], dtype=np.float64)

                # Read in and sum up the local frequencies of the supplied maps.
                for mapfile in self.maps:
                    with h5py.File(mapfile, 'r') as f:
                        hpmap += f['map'][sfreq:efreq, :npol]

        else:
            raise ValueError('No input sky map')

        # initialize angular positions in healpix map
        nside = hp.npix2nside(hpmap.shape[-1])
        telescope._init_trans(nside)
        # rotate the sky map
        hpmap = rot.rotate_map(hpmap, rot=(-rot_ang, 0.0, 0.0))

        # calculate the visibilities
        for fi in range(nfreq):
            bfi = telescope.beam_map(fi)
            vis = np.zeros((npol,) + bfi.shape[:-1], dtype=np.complex128)
            for pi in range(npol):
                vis[pi] = (bfi * hpmap[fi, pi]).sum(axis=-1) * (4 * np.pi / hpmap.shape[-1])

            if self.add_noise:
                vis += telescope.noise(fi)

            with h5py.File('visibilities_%d.hdf5' % fi, 'w') as f:
                f.create_dataset('vis', data=vis)
                f.attrs['add_noise'] = self.add_noise

    def finish(self):
        print 'Generating visibilities done.'
