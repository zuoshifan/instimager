import numpy as np
import h5py

from caput import config
from caput import mpiutil
from caput.pipeline import TaskBase
from caput.pipeline import SingleH5Base
from caput.pipeline import PipelineStopIteration

import Fourier
import rotate as rot


sidereal_day = 23.9344696 * 60 * 60 # Unit: s


class InitUnpolCylinderFFTTelescope(UnpolarisedCylinderFFTTelescope, TaskBase):
    """Initialize an unpolarised cylinder type FFT telescope by reading parameters
    from a YAML configuration file.

    """

    def setup(self):
        print 'Initialize an unpolarised cylinder type FFT telescope.'

    def next(self):
        return self # return a initialized instance

    def finish(self):
        print 'Initialization done.'



# class GenerateVisibility(SingleH5Base):
class GenerateVisibility(TaskBase):
    """Generate simulated visibilities."""

    maps = config.Property(proptype=list, default=[])
    t_obs = config.Property(proptype=float, default=0.0) # Unit: s

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
            with h5py.File(maps[0], 'r') as f:
                mapshape = f['map'].shape

            if lfreq > 0:

                # Allocate array to store the local frequencies
                hpmap = np.zeros((lfreq, npol) + mapshape[2:], dtype=np.float64)

                # Read in and sum up the local frequencies of the supplied maps.
                for mapfile in maps:
                    with h5py.File(mapfile, 'r') as f:
                        hpmap += f['map'][sfreq:efreq, :npol]

        else:
            raise ValueError('No input sky map')

        # rotate the sky map
        hpmap = rot.rotate_map(hpmap, rot=(-rot_ang, 0.0, 0.0))

        # calculate the visibilities
        for fi in range(nfreq):
            bfi = telescope.beam_map(fi)
            vis = np.zeros((bfi.shape[1:]))
            for pi in range(npol):
                vis[pi] = (telescope.beam_map(fi) * hpmap[fi, pi]).sum() * (4 * np.pi / hpmap.shape[-1])

            with h5py.File('visibilities_%d.hdf5' % fi, 'w') as f:
                f.create_dataset('vis', data=vis)
