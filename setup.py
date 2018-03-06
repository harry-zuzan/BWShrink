from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

numpy_inc = numpy.get_include()

extensions = [
	Extension("bws.mrf_wavelet_icm", ["bws/mrf_wavelet_icm.pyx"],
		include_dirs = [numpy_inc],
		),
	Extension("bws.mrf_parallel_icm", ["bws/mrf_parallel_icm.pyx"],
		include_dirs = [numpy_inc],
		extra_compile_args=['-fopenmp'],
		extra_link_args=['-fopenmp'],
		),
	]

setup(
	name = "Bayesian Wavelet Shrinkage",
	ext_modules = cythonize(extensions),
	packages=['bws'],
)
