""" Transfer learning functionality.

Comments:
- Use LoCIT that transfers without considering the labels.
"""

import warnings
import numpy as np

import config as cfg
from classifier import Classifier

import os
import pathlib
current = pathlib.Path().parent.absolute()
p =  os.path.join(current, "src", "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)

def get_transfer_classifier(transfer_function, modus, transfer_settings={}):
    if len(transfer_settings) == 0:
        transfer_settings = initialize_transfer_settings(transfer_function)
    if transfer_function == 'none':
        return NoTransferClassifier(modus)
    else:
        raise ValueError('{} is not in [locit, coral, pwmslt, none]'.format(transfer_function))


def initialize_transfer_settings(transfer_function):
    settings = {'locit': cfg.locit_settings,
        'coral': cfg.coral_settings,
        'pwmstl': cfg.pwmstl_settings,
        'none': None}
    return settings[transfer_function.lower()]



class NoTransferClassifier:

    def __init__(self, modus='anomaly'):
        super().__init__()

        self.classifier = None
        self.modus = modus

    def apply_transfer(self, data=None):
        pass

    def fit_all(self, data,probs, ignore_unchanged=False):
        '''for key in data.keys_:
            y = data.get_domain_labels(key)
            nl = np.count_nonzero(y)
            if not(ignore_unchanged) or (nl > data.get_processed_label_count(key)):
                X = data.get_domain(key)
                clf = Classifier(modus=self.modus)
                clf.fit(X, y)
                self.classifiers[key] = clf
            data.set_processed_label_count(key, nl)'''
        y = np.zeros((0))
        n_features = data.get_domain(0).shape[1]
        X = np.zeros((0,n_features))
        noChange = True
        for key in data.keys_:
            y = np.concatenate((y,data.get_domain_labels(key)))
            X = np.concatenate((X,data.get_domain(key)))
            nl = np.count_nonzero(data.get_domain_labels(key))
            if not(ignore_unchanged) or (nl > data.get_processed_label_count(key)):
                noChange = False
            data.set_processed_label_count(key, nl)
        # Train whole new classifier? Or old classifier with one training instance?
        if (noChange == False):
            clf = Classifier(modus=self.modus)
            clf.fit(X, y,prior= probs)
            self.classifier = clf

    def predict(self, train_key, X, probs, probabilities=False):
        if probabilities:
            scores = self.classifier.predict_proba(X)
            probabs = scores/self.classifier._threshold(prior = probs)
            probabs = np.power(probabs,2)
            xx = 1-np.power(2, -probabs)
            return xx
        else:
            return self.classifier.predict(X)

