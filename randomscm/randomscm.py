from sklearn.base import ClassifierMixin
from sklearn.ensemble import BaseEnsemble
from pyscm import SetCoveringMachineClassifier
from pyscm.rules import DecisionStump

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, check_random_state
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import accuracy_score
import numbers
import itertools
import numpy as np
from warnings import warn
from joblib import effective_n_jobs, Parallel, delayed

MAX_INT = np.iinfo(np.int32).max

class FakeEstim():
    """
    Fake estimator used as a decoy if there is only one class in a specific sub-sampling.
    Does nothing, and has no impact on feature importance.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        self.unique_class = y[0]
        return self

    def predict(self, X):
        return np.array([self.unique_class for _ in range(X.shape[0])])


def _parallel_build_estimators(idx, ensemble, p_of_estims, seeds, X, y, tiebreaker):
    """
    Fit SCM estimators on subsamples of the training data
    """
    estimators = []
    estim_features = []
    for k in idx:
        p_param = p_of_estims[k] # p param for the classifier to fit
        random_state = seeds[k]
        estim = SetCoveringMachineClassifier(p=p_param, max_rules=ensemble.max_rules, model_type=ensemble.model_type, random_state=random_state)
        feature_indices = sample_without_replacement(ensemble._pop_features, ensemble._max_features, random_state=random_state)
        samples_indices = sample_without_replacement(ensemble._pop_samples, ensemble._max_samples, random_state=random_state)
        Xk = (X[samples_indices])[:, feature_indices]
        yk = y[samples_indices]
        if len(list(set(yk))) < 2:
            estim = FakeEstim()
            # raise ValueError("One of the subsamples contains elements from only 1 class, try increase max_samples value")
        if tiebreaker is None:
            estim.fit(Xk, yk)
        else:
            estim.fit(Xk, yk, tiebreaker)
        estim_features.append(feature_indices)
        estimators.append(estim)
    return estimators, estim_features


def _parallel_predict_proba(ensemble, X, idx, results):
    """
    Compute predictions of SCM estimators
    """
    for k in idx:
        res = ensemble.estimators[k].predict(X[:, ensemble.estim_features[k]])
        results = results + res
    return results


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(effective_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = np.full(n_jobs, n_estimators // n_jobs,
                                   dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


class RandomScmClassifier(BaseEnsemble, ClassifierMixin):
    """A Bagging classifier for SetCoveringMachineClassifier()
    The base estimators are built on subsets of both samples
    and features.
    Parameters
    ----------
    n_estimators : int, default=10
        The number of base estimators in the ensemble.
    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator without
        replacement.
        Can be 'sqrt', 'log2', 'auto', None, int or float
        - If int, then draw 'max_samples' samples.
        - If float, then draw 'max_samples * X.shape[0]' samples.
    max_features : int or float, default='sqrt'
        The number of features to draw from X to train each base estimator (
        without replacement.
        Can be 'sqrt', 'log2', 'auto', None, int or float
        - If int, then draw 'max_features' features.
        - If float, then draw 'max_features * X.shape[1]' features.
    max_rules : int
        maximal number of rules for the scm estimators
    p_options : list of float with len =< n_estimators, default=[1.0]
        The estimators will be fitted with values of p found in p_options
        let k be k = n_estimators/len(p_options),
        the k first estimators will have p=p_options[0],
        the next k estimators will have p=p_options[1] and so on...
    model_type : string, default='conjunction'
        type of estimators to build
        accepted values : 'conjunction', 'disjunction'
    n_jobs : int, default=None
        The number of jobs to run in parallel for both fit and
        predict. 'None' means 1. '-1' means using all
        processors.
    random_state : int or RandomState, default=None
        Controls the random resampling of the original dataset
        (sample wise and feature wise).
        If the base estimator accepts a `random_state` attribute, a different
        seed is generated for each instance in the ensemble.
        Pass an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Attributes
    ----------
    n_features_ : int
        The number of features when :meth:`fit` is performed.
    estimators_ : list of estimators
        The collection of fitted base estimators.
    estim_features : list of arrays
        The subset of drawn features for each base estimator.

    Examples
    --------
    >>> random_scm = RandomScmClassifier(p_options=[2, 4], max_samples=0.5, max_features = 0.7)
    >>> random_scm.fit(X_train, y_train)
    >>> hyperparams = random_scm.get_hyperparams()
    >>> importances = random_scm.features_importance()
    >>> disagree = random_scm.classifiers_disagreement(X)

    References
    ----------
    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.
    .. [2] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.
    """
    def __init__(self,
                 n_estimators=100,
                 max_samples=1.0,
                 max_features='sqrt',
                 max_rules=10,
                 p_options=[1.0],
                 model_type="conjunction",
                 n_jobs=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_rules = max_rules
        self.p_options = p_options
        self.model_type = model_type
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.labels_to_binary = {}
        self.binary_to_labels = {}

    def p_for_estimators(self):
        """Return the value of p for each estimator to fit."""
        options_len = len(self.p_options) # number of options
        estims_with_same_p = self.n_estimators // options_len # nb of estimators to fit with the same p
        p_of_estims = []
        if options_len > 1:
            for k in range(options_len - 1):
                opt = self.p_options[k] # an option
                p_of_estims = p_of_estims + ([opt] * estims_with_same_p) # estims_with_same_p estimators with p=opt
        p_of_estims = p_of_estims + ([self.p_options[-1]] * (self.n_estimators - len(p_of_estims)))
        return p_of_estims

    def get_estimators(self):
        """Return the list of estimators of the classifier"""
        if hasattr(self, 'estimators'):
            return self.estimators
        else:
            return "not defined (model not fitted)"

    def get_hyperparams(self):
        """Return the model hyperparameters"""
        hyperparams = {
            'n_estimators' : self.n_estimators, 
            'max_samples' : self.max_samples, 
            'max_features' : self.max_features, 
            'max_rules' : self.max_rules, 
            'p_options' : self.p_options, 
            'model_type' : self.model_type, 
            'random_state' : self.random_state
        }
        return hyperparams

    def labels_conversion(self, labels_list):
        """
        Return the equivalence between labels and binaries
        """
        l = list(set(labels_list))
        labels_dict = {c:idx for idx, c in enumerate(l)}
        if len(l) < 2:
            raise ValueError("Only 1 classe given to the model, needs 2")
        elif len(l) > 2:
             raise ValueError("{} classes were given, multiclass prediction is not implemented".format(len(l)))
        return np.array(l), labels_dict


    def fit(self, X, y, tiebreaker=None):
        """
        Fit the model with the given data
        """
        # Check if 2 classes are inputed and convert labels to binary labels
        X, y = check_X_y(X, y)
        self.classes_, self.labels_to_binary = self.labels_conversion(y)
        self.binary_to_labels = {bin_label:str_label for str_label, bin_label in self.labels_to_binary.items()}
        y = np.array([self.labels_to_binary[l] for l in y])

        # Save the original number of features
        self.n_features = X.shape[1]

        self.estimators = []
        self.estim_features = []
        max_rules = self.max_rules
        p_of_estims_ = self.p_for_estimators()
        model_type = self.model_type

        #seeds for reproductibility
        random_state = self.random_state
        random_state = check_random_state(random_state)
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)
        self._seeds = seeds

        pop_samples, pop_features = X.shape
        max_samples, max_features = self.max_samples, self.max_features

        # validate max_samples
        if max_samples is None:
            max_samples = pop_samples
        elif isinstance(max_samples, str):
            if max_samples == 'auto':
                max_samples = int(np.sqrt(pop_samples))
            elif max_samples == 'sqrt':
                max_samples = int(np.sqrt(pop_samples))
            elif max_samples == 'log2':
                max_samples = int(np.log2(pop_samples))
            else:
                raise ValueError("Invalid value for max_samples: %r" % max_samples)
        elif isinstance(max_samples, float):
            if not (0.0 < max_samples <= 1.0):
                raise ValueError("max_samples float must be in ]0.0, 1.0]")
            max_samples = int(max_samples * pop_samples)
        elif isinstance(max_samples, numbers.Integral):
            if not (0 < max_samples <= pop_samples):
                raise ValueError("max_samples must be in (0, n_samples)")
        # store validated integer row sampling values
        self._max_samples = max_samples
        self._pop_samples = pop_samples

        # validate max_features
        if max_features is None:
            max_features = pop_features
        elif isinstance(max_features, str):
            if max_features == 'auto':
                max_features = int(np.sqrt(pop_features))
            elif max_features == 'sqrt':
                max_features = int(np.sqrt(pop_features))
            elif max_features == 'log2':
                max_features = int(np.log2(pop_features))
            else:
                raise ValueError("Invalid value for max_features: %r" % max_features)
        elif isinstance(max_features, numbers.Integral):
            max_features = max_features
        elif isinstance(max_features, float):
            if not (0.0 < max_features <= 1.0):
                raise ValueError('max_features float must be in ]0.0, 1.0]')
            max_features = round(max_features * pop_features)
        else:
            raise ValueError("max_features must be int or float or None or 'auto' or 'sqrt' or 'log2'")
        if not (0 < max_features <= pop_features):
            raise ValueError("max_features must be in (0, n_features)")
        max_features = max(1, int(max_features))
        # store validated integer feature sampling values
        self._max_features = max_features
        self._pop_features = pop_features

        # parallel loop
        n_jobs, n_estimators_list, starts = _partition_estimators(self.n_estimators, self.n_jobs)
        # building estimators
        all_results = Parallel(n_jobs=n_jobs)(delayed(_parallel_build_estimators)(
                range(starts[i],starts[i+1]), self, p_of_estims_, seeds, X, y, tiebreaker)
            for i in range(n_jobs))

        self.estimators += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estim_features += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        self.feature_importances_ = self.features_importance()

    def predict(self, X):
        """
        Compute model predictions for data in X

        Returns:
        ----------
        predictions : array
            predictions[i] is the predicted class for he sample i in X
        """
        check_is_fitted(self, ["estimators", "estim_features"])
        X = check_array(X)
        predicted_proba = self.predict_proba(X)
        predictions = np.array(np.argmax(predicted_proba, axis=1), dtype=int)
        predictions = np.array([self.binary_to_labels[l] for l in predictions])
        return predictions

    def predict_proba(self, X):
        """
        Predict class probabilities according to the model estimators

        Parameters :
        ----------
        X : array
            a dataset to predict

        Returns : 
        ----------
        proba : array
            proba[c] contains the estimated probability for each sample of X to belong to class c
        """
        check_is_fitted(self, ["estimators", "estim_features"])
        X = check_array(X)
        # parallel loop
        n_jobs, n_estimators_list, starts = _partition_estimators(self.n_estimators, self.n_jobs)
        results = np.zeros(X.shape[0])
        results = Parallel(n_jobs=n_jobs)(delayed(_parallel_predict_proba)(
                self, X, range(starts[i], starts[i+1]), results)
                for i in range(n_jobs))
        votes = sum(results) / self.n_estimators
        proba = np.array([np.array([1 - vote, vote]) for vote in votes])
        return proba

    def features_importance(self):
        """
        Returns an array with the importance value of each feature
        Importance is computed as the ponderated number of use of the feature in the rules
        """
        check_is_fitted(self, ["estimators", "estim_features"])
        importances = np.zeros(self._pop_features)

        for (estim, features_idx) in zip(self.estimators, self.estim_features):
            if isinstance(estim, FakeEstim):
                continue
            # sum the rules importances :
            #rules_importances = estim.get_rules_importances() # activate it when pyscm will implement importance
            rules_importances = np.ones(len(estim.model_.rules)) / len(estim.model_.rules) #delete it when pyscm will implement importance
            for rule, importance in zip(estim.model_.rules, rules_importances):
                global_feat_id = features_idx[rule.feature_idx]
                importances[global_feat_id] += importance
        if sum(importances) > 0:
            importances = importances / sum(importances)
        return importances

    def all_data_tiebreaker(self, model_type, feature_idx, thresholds, rule_type, X, y):
        """
        Choose a rule between rule with equal utility
        Select the one that have the best accuracy if applied alone on all the data

        Parameters :
        ----------
        model_type : strint ('conjunction' or 'dijunction')
            type of the model
        feature_idx, thresholds, rule_type : arrays
            description of the rules
        X, y : arrays
            a dataset used to compare rules

        Returns:
        ----------
        ID of the rule to select
        """
        keep_id = 0
        keep_id_score = -1
        for k in range(len(feature_idx)):
            feat_id, threshold, r_type = feature_idx[k], thresholds[k], rule_type[k]
            stump = DecisionStump(feature_idx=feat_id, threshold=threshold, kind=r_type)
            rule_classif = stump.classify(X).astype('int')
            rule_global_score = (rule_classif == y).sum()
            if rule_global_score > keep_id_score:
                keep_id = k
                keep_id_score = rule_global_score
        return keep_id
    
    def get_estimators_indices(self):
        """
        Get drawn indices along both sample and feature axes
        """
        for seed in self._seeds:
            # operations accessing random_state must be performed identically to those in 'fit'
            feature_indices = sample_without_replacement(self._pop_features, self._max_features, random_state=seed)
            samples_indices = sample_without_replacement(self._pop_samples, self._max_samples, random_state=seed)
            yield samples_indices

    def score(self, X, y):
        check_is_fitted(self, ["estimators", "estim_features"])
        X, y = check_X_y(X, y)
        return accuracy_score(y, self.predict(X))
