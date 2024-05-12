from pyod.models.iforest import IForest
import numpy as np

import os
import pathlib
current = pathlib.Path().parent.absolute()
p =  os.path.join(current, "src", "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)

class IForestWrap(IForest):
    def __init__(self, 
                 n_estimators: int = 100, 
                 max_samples: str = "auto", 
                 contamination: float = 0.1, 
                 max_features: float = 1, 
                 bootstrap: bool = False, 
                 n_jobs: int = 1, 
                 behaviour: str = 'old', 
                 random_state = None, 
                 verbose: int = 0):
        super().__init__(n_estimators, max_samples, contamination, max_features, bootstrap, n_jobs, behaviour, random_state, verbose)

    def fit(self, X, sample_weight=None):
        super().fit(X)

    