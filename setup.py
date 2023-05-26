"""
    randomscm -- The bootstrap aggregating version of the great SCM
    Copyright (C) 2020 Thibaud Godon

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    For a copy of the GNU General Public License, see
         <http://www.gnu.org/licenses/>.

    Contributors:
    ------------
    * Thibaud Godon <thibaud.godon@protonmail.com>
    * Elina Francovic-Fontaine <elina.francovic-fontaine.1@ulaval.ca>
    * Baptiste Bauvin <baptiste.bauvin@lis-lab.fr >

"""
#from numpy import get_include as get_numpy_include
from platform import system as get_os_name
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext

# Configure the compiler based on the OS
if get_os_name().lower() == "darwin":
    os_compile_flags = ["-mmacosx-version-min=10.9"]
else:
    os_compile_flags = []


# Required for the automatic installation of numpy
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


#solver_module = Extension(
#    "pyscm._scm_utility",
#    language="c++",
#    sources=["cpp_extensions/utility_python_bindings.cpp", "cpp_extensions/solver.cpp"],
#    extra_compile_args=["-std=c++0x"] + os_compile_flags,
#)

dependencies = ['scikit-learn>=0.19', 'numpy', 'scipy', 'pyscm-ml']

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="randomscm",
    version="0.1.0",
    packages=find_packages(),
    cmdclass={"build_ext": build_ext},
    setup_requires=dependencies,
    install_requires=dependencies,
    author="Thibaud Godon",
    author_email="thibaud.godon@protonmail.com",
    maintainer="Thibaud Godon",
    maintainer_email="thibaud.godon@protonmail.com",
    description="The Bootstrap aggregating version of the great SCM",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL-3",
    keywords="machine-learning classification ensemble-learning set-covering-machine rule-based-models",
    url="https://github.com/aldro61/pyscm",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    #ext_modules=[solver_module],
    test_suite="nose.collector",
    tests_require=["nose"],
)