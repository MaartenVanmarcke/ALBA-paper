""" Classifier class + classification functionality. """

import numpy as np

from scipy.stats import binom
from functools import partial
from sklearn.metrics import pairwise_distances
from sklearn.svm import SVC
from anomatools.models import SSDO
from IForestWrap import IForestWrap

import os
import pathlib
current = pathlib.Path().parent.absolute()
p =  os.path.join(current, "src", "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)

# -----------------------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------------------

class Classifier:

    # TODO: classification: possibly implement tuning of the SVC
    # TODO: anomaly detection: implement contamination factor correctly
    # TODO: anomaly detection: better hyperparameters?

    def __init__(self, modus='anomaly', contamination=0.1, tol=1e-8, verbose=False):

        self.clf = None
        self.modus_ = str(modus).lower()
        self.c_ = float(contamination)
        self.verbose_ = bool(verbose)
        self.tol_ = float(tol)

        # modus
        if self.modus_ == 'anomaly':
            self.fit = partial(self._anomaly_fit)
            self.predict = partial(self._anomaly_predict)
            self.predict_proba = partial(self._anomaly_predict_proba)
        elif self.modus_ == 'classification':
            self.fit = partial(self._classification_fit)
            self.predict = partial(self._classification_predict)
            self.predict_proba = partial(self._classification_predict_proba)
        else:
            raise ValueError('{} nor know, pick [anomaly, classification]', self._modus)

    # CLASSIFICATION
    def _classification_fit(self, X, y=None, w=None, tuning=False):
        if y is None:
            y = np.zeros(len(X), dtype=int)
        # gamma = median of the pairwise distances in the data
        # speed-up by considering only uniformly drawn subsample of the data: use 20%
        # even with 100 %, 4000 samples with 200 features, the whole thing runs in < sec
        n, _ = X.shape
        rixs1 = np.random.choice(np.arange(0, n, 1), int(n * 0.2), replace=False)
        rixs2 = np.random.choice(np.arange(0, n, 1), int(n * 0.2), replace=False)
        D = pairwise_distances(X[rixs1, :], X[rixs2, :], metric='euclidean')
        gamma = np.median(D.flatten())
        # only use the labeled data to train the SVC
        ixl = np.where(y != 0)[0]
        Xtr = X[ixl, :]
        ytr = y[ixl, :]
        # train the SVC: probability needed, random_state does not matter too much
        self.clf = SVC(C=1.0, kernel='rbf', gamma=gamma, probability=True, class_weight='balanced')
        self.clf.fit(Xtr, ytr, sample_weight=w)
        return self

    def _classification_predict_proba(self, X):
        probabilities = self.clf.predict_proba(X)
        # make sure classes in the right order!
        c = list(self.clf.classes_)
        ixc1 = c.index(-1)
        ixc2 = c.index(1)
        probabilities = probabilities[:, [ixc1, ixc2]]
        return probabilities

    def _classification_predict(self, X):
        predictions = self.clf.predict(X)
        return predictions

    # ANOMALY DETECTION
    def _anomaly_fit(self, X, y=None, w=None, prior = None ):
        if prior == None:
            raise NotImplementedError("You need to specify a prior: loglike or IF")
        if y is None:
            y = np.zeros(len(X), dtype=int)
        # Isolation Forest prior
        # TODO: random_state=i
        if prior == "IF":
            prior = IForestWrap(n_estimators=200, contamination=self.c_, random_state = np.random.randint(0,10000))
            prior.fit(X)
            ss = prior.decision_function(X)
            self.minim, self.maxim = np.min(ss), np.max(ss)

        if prior == "loglike":
            prior = newAnomalyDetectorNorm()
            prior.fit(X)
        #train_prior = prior.decision_scores_
        #test_prior = prior.decision_function(X)
        
        # SSDO
        detector = SSDO(base_detector=prior, k=7)  ## TODO: change k
        detector.fit(X, y)
        self.clf = {0: prior, 1: detector}
        return self

    def _anomaly_predict_proba(self, X):
        # iforest prior
        #test_prior = self.clf[0].decision_function(X)
        # SSDO: probabilities [0: normal, 1: anomaly]
        probabilities = self.clf[1].predict_proba(X)[:,1]
        self.probabs = probabilities
        return probabilities

    def _anomaly_predict(self, X):
        # iforest prior
        #test_prior = self.clf[0].decision_function(X)
        # SSDO: [-1: normal, 1: anomaly]
        predictions = self.clf[1].predict(X)
        return predictions
    
    def _threshold(self, prior = None ):
        if prior == None:
            raise NotImplementedError("You need to specify a prior: loglike or IF")
        idxs = np.flip(np.argsort(self.probabs))
        self.cont_factor = .1
        idx1 = int(np.floor(len(idxs)*self.cont_factor))
        self.threshold = np.mean(self.probabs[idxs[idx1:idx1+1]])
        return self.threshold
        """if prior == "loglike":
            return self.clf[0]._threshold()
        if prior == "IF":
            return (self.clf[0].threshold_-self.minim)/(self.maxim-self.minim)"""
    
    def _decision_function(self, X):
        try:
            return self.clf[0].decision_function(X)
        except Exception:
            return self.clf[0].predict_proba(X)[:,1]


# -----------------------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------------------

def exceed(train_scores, test_scores, prediction=np.array([]), contamination=0.1):
    
    """
    Estimate the example-wise confidence according to the model ExCeeD provided in the paper.
    First, this method estimates the outlier probability through a Bayesian approach.
    Second, it computes the example-wise confidence by simulating to draw n other examples from the population.
    Parameters
    ----------
    train_scores   : list of shape (n_train,) containing the anomaly scores of the training set (by selected model).
    test_scores    : list of shape (n_test,) containing the anomaly scores of the test set (by selected model).
    prediction     : list of shape (n_test,) assuming 1 if the example has been classified as anomaly, 0 as normal.
    contamination  : float regarding the expected proportion of anomalies in the training set. It is the contamination factor.
    Returns
    ----------
    exWise_conf    : np.array of shape (n_test,) with the example-wise confidence for all the examples in the test set.
    
    """
    
    n = len(train_scores)
    t = len(test_scores)
    n_anom = np.int(n*contamination) #expected anomalies
    
    count_instances = np.vectorize(lambda x: np.count_nonzero(train_scores <= x)) 
    n_instances = count_instances(test_scores)

    prob_func = np.vectorize(lambda x: (1+x)/(2+n)) 
    posterior_prob = prob_func(n_instances) #Outlier probability according to ExCeeD
    
    conf_func = np.vectorize(lambda p: 1 - binom.cdf(n - n_anom, n, p))
    exWise_conf = conf_func(posterior_prob)
    #np.place(exWise_conf, prediction == 0, 1 - exWise_conf[prediction == 0]) # if the example is classified as normal,
                                                                             # use 1 - confidence.

    # 2D array (1st column = normal confidences, 2nd column = anomaly confidences)
    confidences = np.vstack((1.0 - exWise_conf, exWise_conf)).T

    return confidences

class newAnomalyDetectorNorm():
    def __init__(self) -> None:
        pass

    def fit(self, X, sample_weight = None):
        pass

    """def _threshold(self):
        return self.threshold"""

    def predict_proba(self, X):
        probabs = np.power(np.linalg.norm(X, axis = 1),2)
        probabs = probabs/np.max(probabs)
        res = np.zeros((len(X),2))
        res[:,1]=probabs
        res[:,0]=1-probabs
        """
        idxs = np.flip(np.argsort(probabs))
        self.cont_factor = .1
        idx1 = int(np.floor(len(idxs)*self.cont_factor))
        self.threshold = np.mean(probabs[idxs[idx1:idx1+1]])"""
        return res
    