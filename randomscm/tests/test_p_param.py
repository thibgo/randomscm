# run this test with : python -m unittest randomscm/tests/test_p_param.py 

import numpy as np
from sklearn.model_selection import GridSearchCV
from unittest import TestCase
from ..randomscm import RandomScmClassifier

class ParamPTests(TestCase):

    def test_p_param(self):
        """
        Test p and p_options parameter values
        """
        n_samples = 250
        X = np.random.rand(n_samples,800) # (samples, features)
        y = np.random.randint(2, size=n_samples)

        try:
            model = RandomScmClassifier(n_estimators=5, p=1.0)
            model.fit(X, y)
            assert model.get_hyperparams['p'] == 1.0
            assert model.p_for_estimators() == [1.0, 1.0, 1.0, 1.0, 1.0]

            model = RandomScmClassifier(n_estimators=5, p=1)
            model.fit(X, y)
            assert model.get_hyperparams['p'] == 1.0
            assert model.p_for_estimators() == [1.0, 1.0, 1.0, 1.0, 1.0]

            model = RandomScmClassifier(n_estimators=5, p=None)
            model.fit(X, y)
            assert model.get_hyperparams['p'] == 1.0
            assert model.p_for_estimators() == [1.0, 1.0, 1.0, 1.0, 1.0]

            model = RandomScmClassifier(n_estimators=5, p=0.1)
            model.fit(X, y)
            assert model.get_hyperparams['p'] == 0.1
            assert model.p_for_estimators() == [0.1, 0.1, 0.1, 0.1, 0.1]
        
            model = RandomScmClassifier(n_estimators=5, p=10)
            model.fit(X, y)
            assert model.get_hyperparams['p'] == 10
            assert model.p_for_estimators() == [10, 10, 10, 10, 10]

        except Exception as e:
            self.fail("error with p parameter calculation when a float or an int is given")
        
        try:
            model = RandomScmClassifier(n_estimators=5, p_options=[1.0])
            model.fit(X, y)
            assert model.get_hyperparams['p'] == [1.0]
            assert model.p_for_estimators() == [1.0, 1.0, 1.0, 1.0, 1.0]
            
            model = RandomScmClassifier(n_estimators=2, p_options=[1.0, 2.0])
            model.fit(X, y)
            assert model.get_hyperparams['p'] == [1.0, 2.0]
            assert model.p_for_estimators() == [1.0, 2.0]
            
            model = RandomScmClassifier(n_estimators=10, p_options=[1.0, 2.0])
            model.fit(X, y)
            assert model.get_hyperparams['p'] == [1.0, 2.0]
            assert model.p_for_estimators() == [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0]

            model = RandomScmClassifier(n_estimators=10, p_options=[1, 2.0])
            model.fit(X, y)
            assert model.get_hyperparams['p'] == [1, 2.0]
            assert model.p_for_estimators() == [1, 1, 1, 1, 1, 2.0, 2.0, 2.0, 2.0, 2.0]
            
            model = RandomScmClassifier(n_estimators=5, p_options=[0.2, 1.5])
            model.fit(X, y)
            assert model.get_hyperparams['p'] == 0.1
            assert model.p_for_estimators() == [0.2, 0.2, 1.5, 1.5, 1.5]

            model = RandomScmClassifier(n_estimators=1, p_options=[0.2, 1.5])
            model.fit(X, y)
            assert model.get_hyperparams['p'] == [0.2]
            assert model.p_for_estimators() == [0.2]

        except Exception as e:
            self.fail("error with p parameter calculation when a list of values is given")