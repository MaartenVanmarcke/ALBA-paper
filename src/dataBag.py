""" Data object containing the domains. """
import numpy as np

import os
import pathlib
current = pathlib.Path().parent.absolute()
p =  os.path.join(current, "src", "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)

class DataBag:

    def __init__(self, bags, bags_labels, X_inst, y_inst):

        self.bags = bags
        self.bags_labels = bags_labels
        self.X_inst = X_inst
        self.y_inst = list(y_inst)
        self.n = len(self.y_inst)
        self.setLengths()
        self.labeled = np.ones_like(y_inst)

        self.anomalies = {}
        self.normals = {}
        for bag in range(len(bags)):
            self.anomalies[bag] = []
            self.normals[bag] = []
            domain = bags[bag]
            for idx in range(len(domain)):
                if self.isAnomaly(bag, idx):
                    self.anomalies[bag].append(domain[idx])
                else:
                    self.normals[bag].append(domain[idx])
            self.anomalies[bag] = np.asarray(self.anomalies[bag])
            self.normals[bag] = np.asarray(self.normals[bag])

    def getNormals(self, bag):
        return self.normals[bag]
    
    def getAnomalies(self, bag):
        return self.anomalies[bag]

    def isAnomaly(self, bag, idx):
        return self.y_inst[self.findFullIdx(bag, idx)]==1

    def setLengths(self):
        self.lengths = {}
        for key in range(len(self.bags)):
            self.lengths[key] = self.bags[key][:,:].shape[0]

    def findFullIdx(self, bag, idx):
        k = 0
        index = 0
        while k<bag:
            index += self.lengths[k]
            k += 1
        return (index+idx)
    
    def label(self, bag, idx):
        self.labeled[self.findFullIdx(bag, idx)] = -2

    def isLabeled(self, bag, idx):
        return self.labeled[self.findFullIdx(bag, idx)]==-2

    def measureAccuracy(self, predictions):
        print(predictions)
        cnt = 0
        for key in predictions:
            for idx in range(len(predictions[key])):
                if (self.getLabel(key,idx) == predictions[key][idx]) or (self.isLabeled(key,idx)):
                    cnt += 1
        return (cnt/self.n)
    
    def getLabel(self, bag, idx):
        if (self.isAnomaly(bag, idx)):
            return 1.0
        else:
            return -1.0
        
    def getLengths(self):
        return self.lengths