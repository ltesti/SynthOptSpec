
#!/usr/bin/env python
# coding=utf-8
import numpy as np
from setuptools import Extension, setup
from Cython.Build import cythonize

from SynthOptSpec._version import __version__

# use README as long_description (for PyPI)
try:
    long_description = open("README", "r", encoding="utf-8").read()
except TypeError:
    # under Python 2.7 the `encoding` keyword doesn't exist.
    print(
        "DEPRECATION: Python 2.7 has reached its end-of-life in 2020. "
        "Please upgrade your Python. SynthOptSpec is not meant for python 2 compatibility."
        "This software may not work as intended (or not work at all). "
    )
    long_description = open("README", "r").read()


extensions = [
    Extension("*", ["SynthOptSpec/accel/*.pyx"],
        include_dirs=[np.get_include()]),
]

setup(
    name="SynthOptSpec",
    version=__version__,
    packages=["SynthOptSpec"],
    author="Leonardo Testi",
    author_email="ltesti120a@gmail.com",
    description="Utilities and scripts for the ECOGAL project to format synthetic spectra.",
    long_description=long_description,
    install_requires=["numpy>=1.9", "matplotlib", "cython>=3.0", "astropy>=5.0"],
    #ext_modules=cythonize("SynthOptSpec/utils_c.pyx"),
    #include_dirs=[np.get_include()],
    ext_modules=cythonize(extensions),
    license="LGPLv3",
    url="https://github.com/ltesti/SynthOptSpec",
    download_url="https://github.com/ltesti/SynthOptSpec/archive/{}.tar.gz".format(
        __version__
    ),
    keywords=["science", "astronomy", "optical spectroscopy","Synthetic spectra"],
    classifiers=[
        "Development Status :: 0 - Under Development/Unstable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Programming Language :: Python :: 3",
    ],
)
