# -*- coding: utf-8 -*-
# ######### COPYRIGHT #########
#
# Copyright(c) 2021 Thibaud Godon
# -----------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    license = 'Apache-2.0'
    keywords = ('machine learning, supervised learning, classification, '
                'ensemble methods, bagging')
    packages = find_packages(exclude=['*.tests'])
    install_requires = ['scikit-learn>=0.19', 'numpy', 'scipy', 'cvxopt', 'pyscm-ml']
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
