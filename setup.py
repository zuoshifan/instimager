from setuptools import setup, find_packages
from numpy.distutils.core import Extension
import numpy as np


## Try and decide whether to use Cython to compile the source or not.
try:
    from Cython.Build import cythonize
    HAVE_CYTHON = True
except ImportError as e:
    warnings.warn("Cython not installed.")
    HAVE_CYTHON = False


def cython_file(filename):
    filename = filename + ('.pyx' if HAVE_CYTHON else '.c')
    return filename

## Setup the extensions we are going to build
ft_ext = Extension('imager.fouriertransform', [cython_file('imager/fouriertransform')],
                     include_dirs=['.', np.get_include()],)

## Apply Cython to the extensions if it's installed
exts = [ft_ext]
if HAVE_CYTHON:
    exts = cythonize(exts, include_path=['.', np.get_include()])


setup(
    name = 'instimager',
    version = 0.1,

    packages = find_packages(),
    # scripts=['scripts/caput-pipeline'],
    requires = ['numpy', 'h5py', 'healpy'],  # Probably should change this.

    # metadata for upload to PyPI
    author = "Shifan Zuo",
    author_email = "sfzuo@bao.ac.cn",
    description = "Instantaneous imager for FFT telescope.",
    license = "GPL v3.0",
    url = "https://github.com/zuoshifan/instimager",
    ext_modules=exts,
)
