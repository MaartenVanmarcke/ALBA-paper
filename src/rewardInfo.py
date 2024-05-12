""" Data object containing the domains. """

import numpy as np
from collections import OrderedDict

import os
import pathlib
current = pathlib.Path().parent.absolute()
p =  os.path.join(current, "src", "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)

class RewardInfo:

    # TODO: implement copy functionality
    # TODO: additional functionality as needed

    def __init__(self, nbags: int):

        self.rewards = {}
        for bag in range(nbags):
            self.rewards[bag] = []

        self.banditRewards = {}
        for bag in range(nbags):
            self.banditRewards[bag] = []

        self.chosenArms = {}
        for bag in range(nbags):
            self.chosenArms[bag] = [0]

        self.inRoundRobin = True
        self.RoundRobinEndIteration = -1

        self.aucs = {}

    def setIterationEndRoundRobin(self, iteration:int):
        self.RoundRobinEndIteration = iteration

    def getIterationEndRoundRobin(self):
        return self.RoundRobinEndIteration
    
    def isInRoundRobin(self):
        return self.inRoundRobin

    def endRoundRobin(self):
        self.inRoundRobin = False

    def startRoundRobin(self):
        self.inRoundRobin = True

    # SETTERS
    def updateReward(self, bag: int, reward: float):
        if (reward != None):
            self.rewards[bag].append(reward)

    def updateAllRewards(self, rewards: [float]):
        if (len(rewards)!= len(self.rewards.keys())):
            raise Exception("Unvalid argument")
        for bag in range(rewards):
            self.updateReward(bag, rewards[bag])

    def getRewards(self, bag:int):
        return self.rewards[bag]
    
    def getLastReward(self, bag:int):
        return self.getRewards(bag)[-1]
    
    def getAllRewards(self):
        return self.rewards
    
    def updateBanditReward(self, bag: int, reward: float):
        if (reward != None):
            self.banditRewards[bag].append(reward)

    def updateAllBanditRewards(self, rewards: dict):
        if (len(rewards.keys())!= len(self.rewards.keys())):
            raise Exception("Unvalid argument")
        for bag in range(len(rewards.keys())):
            self.updateBanditReward(bag, rewards[bag])

    def updateAuc(self, name: str, auc: float):
        if (name not in self.aucs.keys()):
            self.aucs[name] = [auc]
        else:
            self.aucs[name].append(auc)

    def getAUC(self):
        return self.aucs

    def getBanditRewards(self, bag:int):
        return self.banditRewards[bag]
    
    def getLastBanditReward(self, bag:int):
        return self.getBanditRewards(bag)[-1]
    
    def getAllBanditRewards(self):
        return self.banditRewards
    
    def chooseArm(self, bag):
        for key in self.chosenArms.keys():
            cnt = self.chosenArms[key][-1]
            if key == bag:
                self.chosenArms[key].append(cnt+1)
            else:
                self.chosenArms[key].append(cnt)
    
    def getChosenArm(self, bag):
        return self.chosenArms[bag]
    
    def getAllChosenArms(self):
        return self.chosenArms
