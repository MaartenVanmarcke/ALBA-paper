import math, time, sys
import numpy as np
from collections import OrderedDict

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances

from active_learning import active_learning
from bandit import MAB, DomainArm
from classifier import Classifier
from transfer_learning import get_transfer_classifier
from rewardInfo import RewardInfo
from data import Data

# -----------------------------------------------------------------------------------------
# Separate detector baseline
# -----------------------------------------------------------------------------------------

""" 1. has transfer
    2. has domain selection
    3. has instance selection

Description:
------------
MAB strategies for multi-domain active learning. Without transfer.

Parameters:
-----------
1. Which reward function to use for assessing the impact of labeling? --> "mabreward"
    - entropy decrease
    - label flips
    - cosine
2. How to estimate the reward for each armed-bandit? --> "mab"
    - the classic MAB strategies
3. Which level of abstraction to choose each armed bandit? --> "abstraction_level" / "abstraction_strat"
    - 0 = each domain is an armed bandit
    - >0 = divide each domain into clusters
    - smart or naive strategy (i.e., decide yourself how to divide the clusters between domains)
4. How to select a point within each armed bandit? --> "al_strategy"
    - entropy
    - random
"""


class BaselineALMethod:
    def _flatten(data, data_labels):
        D = np.zeros((0,2))
        for key in data.keys():
            D = np.concatenate((D, data[key]))
        return {0:D}, [int(len(data_labels[data_labels==1])>0)]