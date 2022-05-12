# sirius

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://badge.fury.io/py/sirius.svg)](https://badge.fury.io/py/sirius)
[![tests](https://github.com/casangi/sirius/actions/workflows/python-testing.yml/badge.svg?branch=main)](https://github.com/casangi/sirius/actions/workflows/python-testing.yml)

SiRIUS is a component list radio interferometry visibilities simulator in development for the VLA, ALMA, and the ngVLA telescopes. It makes use of modular Common Astronomy Software Applications ([CASA](https://casa.nrao.edu/)), the CASA Next Generation Infrastructure framework ([CNGI](https://github.com/casangi/cngi_prototype)), and dask-ms ([dask-ms](https://github.com/ratt-ru/dask-ms)).

> 📝 SiRIUS is under active development! Breaking API changes are still happening on a regular basis, so proceed with caution.

# Installing
It is recommended to use the [conda](https://docs.conda.io/projects/conda/en/latest/) environment manager to create a clean, self-contained runtime where sirius and all its dependencies can be installed:
```sh
conda create --name sirius python=3.8 --no-default-packages
conda activate sirius

```
> 📝 On macOS it is required to pre-install `python-casacore` using `conda install -c conda-forge python-casacore`. After this is fixed upstream, documentation for installing sirius using `virtualenv` will be added here.

Making sirius available for download from conda-forge directly is pending, so until then the current recommendation is to sully that pristine environment by calling pip [from within conda](https://www.anaconda.com/blog/using-pip-in-a-conda-environment), like this:
```sh
pip install sirius
```
The basic dependency versions fixed to a given release are frozen using `pip list --format=freeze > requirements.txt`.

## Developer installation

Developers build and install sirius from their local clones of the source code. With their base conda environment active,
`pip install -e .[complete]`. Specifying the `[complete]` install will download extras, including [pre-commit](https://pre-commit.com/) (required for committing code changes that pass the linter) and [pytest](https://docs.pytest.org/en/7.1.x/) (which is helpful for running unit tests locally).
In addition to development extras, the packages required to build the documentation can be installed locally from source using
```sh
pip install -e .[docs]
```
The complete dependency versions fixed to a given release are frozen using `pip list --format=freeze > requirements-dev.txt`.

It is also possible to download directly from the requirements files using `pip install -r requirements.txt` or `pip install -r requirements-dev.txt`, just be aware that it might not be confirmed to be compatible with bleeding edge development between release tags.

Before committing to a branch pre-commit should be installed:
```sh
conda install -c conda-forge pre-commit
cd sirius #source directory
pre-commit install
```
