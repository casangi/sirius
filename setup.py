from setuptools import setup, find_packages

with open('README.md', "r") as fid:   #encoding='utf-8'
    long_description = fid.read()
    
#Load version. sirius/_version.py is the canonical place where the version number is stored.
import re
VERSIONFILE="sirius/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))
################

setup(
    name='sirius',
    version=verstr,
    description='Simulation of Radio Interferometry from Unique Sources',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='National Radio Astronomy Observatory',
    author_email='casa-feedback@nrao.edu',
    url='https://github.com/casangi/sirius',
    license='Apache-2.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['bokeh>=2.4.2',
                      'numba-scipy>=0.3.0',
                      'dask>=2022.01.0',
                      'distributed>=2022.01.0',
                      'graphviz>=0.19.1',
                      'matplotlib>=3.1.2',
                      'numba>=0.55.1',
                      'numcodecs>=0.9.1',
                      'numpy>=1.21.5',
                      'pandas>=0.25.2',
                      'scipy>=1.4.1',
                      'scikit-learn>=0.22.2',
                      'toolz>=0.10.0',
                      'xarray>=0.20.2',
                      'zarr>=2.10.3',
                      'fsspec>=2022.1.0',
                      'gdown>=3.12.2',
                      'pytest>=6.2.4',
                      'ipympl>=0.7.0',
                      'dask-ms>=0.2.6',
                      'casaconfig>=0.0.27',
                      'casatools>=6.3.0.48',
                      'casatasks>=6.3.0.48',
                      'casadata>=2021.8.23',
                      'python-casacore>=3.4.0',
                      'astropy>=4.3.1',
                      'cngi-prototype>=1.0.1'
                      ],
    extras_require={
        'dev': [
            'pytest>=5.3.5',
            'black>=19.10.b0',
            'flake8>=3.7.9',
            'isort>=4.3.21',
            's3fs>=0.4.0',
            'pylint>=2.4.4',
            'radio-telescope-delay-model>=0.0.3',
            #'pytest-pep8',
            #'pytest-cov'
        ]
    }

)
