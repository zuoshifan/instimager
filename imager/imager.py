import numpy as np
import healpy as hp
import h5py

import fourier
import rotate as rot

from caput import config
from caput import mpiutil
from caput.pipeline import TaskBase
# from caput.pipeline import SingleH5Base
from caput.pipeline import PipelineStopIteration



sidereal_day = 23.9344696 * 60 * 60 # Unit: s


class InitUnpolCylinderFFTTelescope(fourier.UnpolarisedCylinderFFTTelescope, TaskBase):
# class InitUnpolCylinderFFTTelescope(TaskBase, fourier.UnpolarisedCylinderFFTTelescope):
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
    output_file = config.Property(proptype=str, default='')

    def setup(self):
        print 'Begin to generate simulated visibilities.'

    def next(self, telescope):
        nfreq = telescope.nfreq
        npol = telescope.num_pol_sky

        # the earth rotation angle, equivalently the sky rotates a negative rot_ang
        rot_ang = 360.0 * self.t_obs / sidereal_day # degree
        telescope.rot_ang = rot_ang # record rotation angle so can pass to next task

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

            print 'bfi.shape: ', bfi.shape
            print 'zenith: ', telescope.zenith
            with h5py.File('bfi_%d.hdf5' % fi, 'w') as f:
                f.create_dataset('map', data=bfi[0, 0])

            vis = np.zeros((npol,) + bfi.shape[:-1], dtype=np.complex128)
            for pi in range(npol):

                with h5py.File('bfihp_%d_%d.hdf5' % (fi, pi), 'w') as f:
                    print 'shape: ', (bfi * hpmap[fi, pi]).shape
                    f.create_dataset('map', data = (bfi * hpmap[fi, pi])[0, 0])

                vis[pi] = (bfi * hpmap[fi, pi]).sum(axis=-1) * (4 * np.pi / hpmap.shape[-1])

            if self.add_noise:
                vis += telescope.noise(fi)

            with h5py.File('visibilities_%d.hdf5' % fi, 'w') as f:
                f.create_dataset('vis', data=vis)
                f.attrs['add_noise'] = self.add_noise
                f.attrs['f_index'] = fi

        return telescope

    def finish(self):
        print 'Generating visibilities done.'



class FFTMapMaking(TaskBase):
    """Map-making via Fast Fourier Transform the gridded visibilities."""

    vis_files = config.Property(proptype=list, default=[])

    def setup(self):
        print 'Start to make sky maps.'

    def next(self, telescope):
        for vis_file in self.vis_files:
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

            with h5py.File('beamprod_%d.hdf5' % fi, 'w') as f:
                f.create_dataset('map', data=beam_prod)
            with h5py.File('Tgrid_%d.hdf5' % fi, 'w') as f:
                f.create_dataset('map', data=T_grid)

            # convert to healpix map
            T_map = np.zeros((telescope.num_pol_sky, 12 * telescope._nside**2), dtype=T_grid.dtype)
            # T_map[..., telescope.hp_pix(fi)] = T_grid.flatten()
            T_map[..., telescope.hp_pix(fi)] = T_grid

            # inversely rotate the sky map
            T_map = rot.rotate_map(T_map, rot=(telescope.rot_ang, 0.0, 0.0))

            with h5py.File('map_%d.hdf5' % fi, 'w') as f:
                f.create_dataset('map', data=T_map)

    def finish(self):
        print 'Making sky maps done.'
