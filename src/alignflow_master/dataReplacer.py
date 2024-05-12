"""Some class used for data manipulation for the normalizing flows."""

import numpy as np

class DataReplacer():

    def __init__(self, num_sources) -> None:
        self.num_sources = num_sources
        self.data = {}
        self.latent = {}


    def removeLatent(self):
        self.latent = {}

    def removeWeights(self):
        for key in range(self.num_sources):
            self.data[key][:,-1] = np.ones_like(self.data[key][:,-1], dtype = np.float32)

    def setInitData(self, data):
        self.data = {}
        for key in range(self.num_sources):
            self.data[key] = data[key].astype(np.float32)
            self.data[key] = np.hstack((self.data[key], np.zeros((len(self.data[key][:,0]),1),dtype = np.float32)))
        return self

    def setData(self, data):
        for key in range(self.num_sources):
            self.data[key][:,:-1] = data[key].astype(np.float32)

    def setWeights(self, weights):
        self.removeWeights()
        for key in range(self.num_sources):
            self.data[key][:,-1] = weights[key]

    def setLatent(self, latents):
        self.removeLatent()
        for key in range(self.num_sources):
            self.latent[key] = latents[key]


    def getData(self):
        return self.data
    
    def getWeights(self):
        D = {}
        for key in range(self.num_sources):
            D[key] = self.data[key][:,-1]
        return D
    
    def getNumSources(self):
        return self.num_sources
    
    def getLatent(self):
        return self.latent