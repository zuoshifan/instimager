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


class InitUnpolarisedCylinderFourierTransformTelescope(fourier.UnpolarisedCylinderFourierTransformTelescope, TaskBase):
    """Initialize an unpolarised cylinder type FFT telescope by reading parameters
    from a YAML configuration file.

    """

    def setup(self):
        self.i = 0
        if mpiutil.rank0:
            print 'Initialize an unpolarised cylinder type FFT telescope.'

    def next(self):
        if self.i > 0:
            raise PipelineStopIteration()
        self.i += 1
        return self # return a initialized instance

    def finish(self):
        if mpiutil.rank0:
            print 'Initialization done.'



class InitUnpolCylinderFFTTelescope(fourier.UnpolarisedCylinderFFTTelescope, TaskBase):
    """Initialize an unpolarised cylinder type FFT telescope by reading parameters
    from a YAML configuration file.

    """

    def setup(self):
        self.i = 0
        if mpiutil.rank0:
            print 'Initialize an unpolarised cylinder type FFT telescope.'

    def next(self):
        if self.i > 0:
            raise PipelineStopIteration()
        self.i += 1
        return self # return a initialized instance

    def finish(self):
        if mpiutil.rank0:
            print 'Initialization done.'



# class GenerateVisibility(SingleH5Base):
class GenerateVisibility(TaskBase):
    """Generate simulated visibilities."""

    maps = config.Property(proptype=list, default=[])
    t_obs = config.Property(proptype=float, default=0.0) # Unit: s
    add_noise = config.Property(proptype=bool, default=True)
    output_file = config.Property(proptype=str, default='')

    def setup(self):
        if mpiutil.rank0:
            print 'Begin to generate simulated visibilities.'

    def next(self, telescope):

        rot_ang = 360.0 * self.t_obs / sidereal_day # degree

        # generate visibilities
        # the earth rotation angle, equivalently the sky rotates a negative rot_ang
        vis = telescope.gen_visibily(self.maps, rot_ang=-rot_ang, add_noise=self.add_noise)
        with h5py.File(self.output_file, 'w') as f:
            f.create_dataset('vis', data=vis)
            f.attrs['t_obs'] = self.t_obs
            f.attrs['add_noise'] = self.add_noise
            f.attrs['zenith'] = telescope.zenith
            f.create_dataset('baselines', data=telescope.blvector)
        return telescope

    def finish(self):
        if mpiutil.rank0:
            print 'Generating visibilities done.'



class FFTMapMaking(TaskBase):
    """Map-making via Fast Fourier Transform the gridded visibilities."""

    vis_file = config.Property(proptype=str, default='')
    output_file = config.Property(proptype=str, default='')
    dirty_map = config.Property(proptype=bool, default=False) # get dirty map if True

    def setup(self):
        if mpiutil.rank0:
            print 'Start to make sky maps.'

    def next(self, telescope):
        with h5py.File(self.vis_file, 'r') as f:
            vis = f['vis'][...]
            t_obs = f.attrs['t_obs']
            rot_ang = 360.0 * t_obs / sidereal_day # degree

        T_map = telescope.map_making(vis, rot_ang=rot_ang, divide_beam=not(self.dirty_map))

        with h5py.File(self.output_file, 'w') as f:
            f.create_dataset('map', data=T_map)

    def finish(self):
        if mpiutil.rank0:
            print 'Making sky maps done.'
