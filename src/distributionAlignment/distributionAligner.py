"""
This file implements an interface for aligning bags using the normal aligner.

    class NormalAligner
        align(self, bags) -> newBags, newXinst
"""

import statistics
import numpy as np

class DistributionAligner:
    def __init__(self) -> None:
        pass

    def align(self, bags):
        return bags

class NormalAligner(DistributionAligner):
    def __init__(self) -> None:
        super().__init__()

    def align(self, bags):
        newBags = {}
        for key in bags:
            bag = bags[key]
            med = np.zeros(bag[0].shape)
            std = np.zeros(bag[0].shape)
            for i in range(len(bag[0])):
                med[i] = statistics.median(bag[:,i])
                std[i] = statistics.stdev(bag[:,i])
            newBags[key] = (bag-med)/std
        newXinst = np.zeros((0,bag[0].shape[0]))
        for key in bags:
            newXinst = np.concatenate((newXinst, newBags[key]))
        return newBags, newXinst
                
            


'''al = NormalAligner()
x = {0: np.random.random((4,3)),1:np.random.random((5,3))}
print(al.align(x))'''