# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2021
# -----------------
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#
# Contributors:
# ------------
#
# * Thibaud Godon <thibaud.godon@gmail.com>
# * Elina Francovic-Fontaine <elina.francovic-fontaine.1@ulaval.ca>
# * Baptiste Bauvin <baptiste.bauvin.1@ulaval.ca>
#
#
# Description: 
# -----------
# Description: The Bootstrap aggregating version of the great SCM
#
#
# Version:
# -------
# Version: 0.0
#
#
# Licence:
# -------
# License: GPL-3
#
#
# ######### COPYRIGHT #########
import os
from setuptools import setup, find_packages

def setup_package():
    """
    Setup function
    """
    name = 'randomscm'
    version = 0.0
    dir = 'randomscm'
    description = 'The Bootstrap aggregating version of the great SCM'
    here = os.path.abspath(os.path.dirname(__file__))
    url = "https://github.com/thibgo/randomscm"
    project_urls = {
        'Source': url,
        'Tracker': '{}/issues'.format(url)}
    author = 'Thibaud Godon'
    author_email = 'thibaud.godon@gmail.com'
    maintainer = 'Thibaud Godon',
    maintainer_email = 'thibaud.godon@gmail.com',
    license = 'GPL-3'
    keywords = ('machine learning, supervised learning, classification, '
                'ensemble methods, bagging')
    packages = find_packages(exclude=['*.tests'])
    install_requires = ['scikit-learn>=0.19', 'numpy', 'scipy', 'cvxopt', "pyscm"]
    python_requires = '>=3.5'
    extras_require = {}
    include_package_data = True

    setup(name=name,
          version=version,
          description=description,
          url=url,
          project_urls=project_urls,
          author=author,
          author_email=author_email,
          maintainer=maintainer,
          maintainer_email=maintainer_email,
          license=license,
          keywords=keywords,
          packages=packages,
          install_requires=install_requires,
          python_requires=python_requires,
          extras_require=extras_require,
          include_package_data=include_package_data)

if __name__ == "__main__":
    setup_package()
