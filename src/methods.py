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

import os
import pathlib
current = pathlib.Path().parent.absolute()
p =  os.path.join(current, "src", "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)

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


class MABMethod:
    def __init__(
        self,
        modus="anomaly",
        transfer_function="none",
        mab="none",
        mabreward=None,#"entropy",
        mab_alpha=1.0,
        mab_sigma=0.1,
        abstraction_level=0,
        abstraction_strat="naive",
        al_strategy="random",
        query_budget=100,
        verbose=False,
        rewardInfo:RewardInfo=None
    ):

        # general
        self.modus = str(modus).lower()
        self.tf_function = str(transfer_function).lower()

        # MAB specific selectors
        self.mab = str(mab).lower()
        self.mabreward = str(mabreward).lower()
        self.mab_alpha = float(mab_alpha)
        self.mab_sigma = float(mab_sigma)
        self.al_strategy = str(al_strategy).lower()
        self.abstraction_level = int(abstraction_level)
        self.abstraction_strat = str(abstraction_strat).lower()

        # other
        self.iteration_ = 0
        self.classifier = None
        self.bandit = None
        self.i = None
        self.probs0 = None
        self.query_budget = int(query_budget)
        self.verbose = bool(verbose)
        self.rewardInfo = rewardInfo

    ## The algorithm!!
    def fit_query(self, train_data, probs, return_debug_info=False):

        ## Line 5
        # first iteration: divide the domain in clusters (smart or naive strategy)
        if self.iteration_ == 0:
            self._initialize_armed_bandits(train_data)

        ## Line 6 & 13
        # fit the classifier (transfer / no transfer)
        # this is still a single classifier per domain
        if self.iteration_ == 0:
            self.classifier = get_transfer_classifier(
                self.tf_function, self.modus
            )  # could be none
            self.classifier.apply_transfer(train_data)
        self.classifier.fit_all(train_data, probs, ignore_unchanged=False)

        ## QUESTION: where is the for loop?? 
        ##      -> do you take the first query in the variable <queries> and then call fit_query again? Until self.query_budget times? 
        # mab strategy
        ## line 14 & 10: estimate payoffs and thén get order of arms to play
        play_order = self._play_multi_armed_bandit(train_data, probs)

        # instance selection within each domain
        ## line 11
        all_scores = active_learning(
            train_data.keys_, train_data, self.classifier, self.al_strategy, prior= probs
        )

        # sort queries based on the play_order
        queries = []
        for ID in play_order:
            key = self.armed_bandits[ID]["domain"]
            ixs = self.armed_bandits[ID]["indices"]
            # get the corresponding AL scores and sort
            als = all_scores[(all_scores[:, -1] == key)][ixs]
            new_queries = [
                (int(q[2]), int(q[1])) for q in als[als[:, 0].argsort()[::-1]]
            ]
            queries.extend(new_queries)

        # print(queries)

        self.iteration_ += 1
        return queries

    def predict(self, test_data, probs,probabilities=False):
        predictions = OrderedDict({})
        n_features = test_data.get_domain(0).shape[1]
        X = np.zeros((0,n_features))
        nl = np.zeros((1))
        for key in test_data.keys_:
            X = np.concatenate((X,test_data.get_domain(key)))
            nl = np.append(nl, len(list(X)))
        prs = self.classifier.predict(0, X, probs,probabilities)
        for key in test_data.keys_:
            predictions[key] = prs[int(nl[key]):int(nl[key+1])]

        return predictions

    ## Line 5: clustering the data
    def _initialize_armed_bandits(self, train_data):

        # keep track of the armed bandits
        # structure: key = armed bandit ID, value = {domain_key: ..., indices: [...]}
        self.armed_bandits = OrderedDict({})

        ## Each domain is a seperate armed bandit (1 big cluster per domain)
        if self.abstraction_level < 2:
            for ID, key in enumerate(train_data.keys_):
                n, _ = train_data.get_domain_shape(key)
                self.armed_bandits[ID] = {
                    "domain": key,
                    "indices": np.arange(n),
                }
            return

        ## Each domain gets divided in clusters, with a specified number of clusters
        # divide each domain in the given number of clusters
        if self.abstraction_strat == "naive":
            ID = 0
            for _, key in enumerate(train_data.keys_):
                n, _ = train_data.get_domain_shape(key)
                X = train_data.get_domain(key)

                # cluster
                clusterer = KMeans(n_clusters=self.abstraction_level)
                labels = clusterer.fit_predict(X)

                # store label indices
                for ul in np.unique(labels):
                    ixc = np.where(labels == ul)[0]
                    if len(ixc) > 0:
                        self.armed_bandits[ID] = {
                            "domain": key,
                            "indices": ixc,
                        }
                        ID += 1

        ## Each domain gets divided in clusters, with a variable number of clusters
        # smart division in the number of clusters: DBSCAN
        elif self.abstraction_strat == "smart":
            ID = 0
            for _, key in enumerate(train_data.keys_):
                n, _ = train_data.get_domain_shape(key)
                X = train_data.get_domain(key)

                # pairwise distances
                D = pairwise_distances(X)
                Dsorted = np.sort(D)
                eps_est = np.median(Dsorted[:, 10])

                # cluster
                clusterer = DBSCAN(eps=eps_est, min_samples=5, metric="precomputed")
                labels = clusterer.fit_predict(D)

                # store label indices
                for ul in np.unique(labels):
                    ixc = np.where(labels == ul)[0]
                    if len(ixc) > 0:
                        self.armed_bandits[ID] = {
                            "domain": key,
                            "indices": ixc,
                        }
                        ID += 1

        else:
            raise Exception("INPUT: unknown `abstraction_strat`")

    def _play_multi_armed_bandit(self, train_data, probs):

        # TODO: code to deal with very small clusters

        # initialize everything
        if self.iteration_ == 0:
            nb = len(self.armed_bandits)

            # init bandit and arms
            ## Initialize the MAB algorithm
            self.bandit = MAB(
                nb,
                T=self.query_budget,
                solver=self.mab,
                solver_param={"alpha": self.mab_alpha, "sigma": self.mab_sigma},
                rewardInfo=self.rewardInfo
            )
            ## Initialize the reward function. Initialize all on 0 -> line 8
            self.arms = {
                ID: DomainArm(metric=self.mabreward)
                for ID, _ in self.armed_bandits.items()
            }

        # update the reward (first time is handled in the arms itself)
        ## line 14
        all_probs = {}
        all_preds = {}
        n_features = train_data.get_domain(0).shape[1]
        X = np.zeros((0,n_features))
        nl = np.zeros((1))
        for key in train_data.keys_:
            X = np.concatenate((X,train_data.get_domain(key)))
            nl = np.append(nl, len(list(X)))
        probs = self.classifier.predict(0, X, probs, True)
        preds = self.classifier.predict(0, X, probs, False)
        for key in train_data.keys_:
            all_probs[key] = probs[int(nl[key]):int(nl[key+1])].flatten()
            all_preds[key] = preds[int(nl[key]):int(nl[key+1])]

        if self.iteration_ == 0:
            for ID, cluster in self.armed_bandits.items():
                k = cluster["domain"]
                ixs = cluster["indices"]
                self.arms[ID].update_reward(all_probs[k][ixs], all_preds[k][ixs],np.linalg.norm(train_data.get_domain(k)))

        # get reward for the played arm last time
        # this can only start from the second round
        if self.iteration_ > 0:
            last_labeled = train_data.get_last_labeled()
            for ID, cluster in self.armed_bandits.items():
                if last_labeled[0] == cluster["domain"]:
                    if last_labeled[1] in cluster["indices"]:
                        k = cluster["domain"]
                        ixs = cluster["indices"]
                        self.arms[ID].update_reward(all_probs[k][ixs], all_preds[k][ixs],np.linalg.norm(train_data.get_domain(k)))
                        self.i = ID
                        break
            ## Tell from which cluster (arm) you have queried
            self.bandit.play(self.i, self.arms)
            if (self.rewardInfo != None):
                self.rewardInfo.updateReward(self.i, self.arms[self.i].reward)
                try:
                    self.rewardInfo.updateProbsContr(self.i, self.arms[self.i].probsContribution)
                    self.rewardInfo.updateAlignContr(self.i, self.arms[self.i].alignContribution)
                except Exception:
                    pass

        # decide order to play the arms (self.i = ID of the selected arm this round)
        play_order = self.bandit.decide(return_estimate=False)

        # go from play order to the specific arm to play
        # this depends on whether the examples in the domain have already been labeled
        return play_order
