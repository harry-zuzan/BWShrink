from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

numpy_inc = numpy.get_include()

extensions = [
    Extension("bws.mrf_wavelet_icm", ["bws/mrf_wavelet_icm.pyx"],
        include_dirs = [numpy_inc],
		)
	]

setup(
    name = "Bayesian Wavelet Shrinkage",
    ext_modules = cythonize(extensions),
	packages=['bws'],
)
