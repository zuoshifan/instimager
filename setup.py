from setuptools import setup, find_packages


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
    url = "https://github.com/zuoshifan/instimager"
)
