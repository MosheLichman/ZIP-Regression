from Cython.Build import cythonize
from distutils.core import setup
# python setup.py build_ext --inplace
setup(
    ext_modules=cythonize("model/fast_methods.pyx"),
)
