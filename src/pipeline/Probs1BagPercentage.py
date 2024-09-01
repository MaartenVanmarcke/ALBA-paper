from typing import Any
import matplotlib.pyplot as plt
import os
import pathlib
import numpy as np
from scipy.stats import rankdata
current = pathlib.Path().absolute()
current = os.path.join(current, "src")
import numpy as np
p =  os.path.join(current, "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)
import csv

class PlotPerformance:
    def __init__(self, datasets,methods, steps) -> None:
        self.datasets = datasets
        self.methods = methods
        self.steps = steps
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
            namesDatasets = {}
            for i in range(len(self.datasets)):
                namesDatasets[self.datasets[i]] = i
            namesMethods = {}
            for i in range(len(self.methods)):
                namesMethods[self.methods[i]] = i
            #rankstdev = np.zeros((len(self.datasets)))
            for method in range(len(self.methods)):
                probsBag = np.zeros((101,10, len(self.datasets)))
                for filename in os.listdir("bagprobs"):
                    f = os.path.join("bagprobs", filename)
                    # checking if it is a file
                    for ds in range(len(self.datasets)):
                        dataset = self.datasets[ds]
                        if (self.methods[method] in filename
                            and dataset in filename
                            and os.path.isfile(f)):
                            with open(f, 'r') as readFile:
                                reader = csv.reader(readFile)
                                lines = list(reader)
                                for line in range(len(lines)):
                                    probsBag[line,:, ds] = np.asarray([float(ww) for ww in (lines[line])[1:]], dtype = float)

                
                ## CALCULATE PROPORTION
                probsBagIdxs = probsBag == 1
                count = np.count_nonzero(probsBagIdxs)
                totalCount = np.prod(probsBagIdxs.shape)
                proportiton = count/totalCount
                #print(probsBag)
                print(self.methods[method], count, totalCount, proportiton)
                #print(probsBag[-1,:,:])
                """## calculate variance
                rankstdev[:] = np.std(rank, axis = 0)
                print(rankmean)"""
            return

if __name__=="__main__":
    pp = PlotPerformance(["40_Vowels", "47_yeast","20_letter", "12_Fault",  "41_Waveform"],#["29_Pima_equal_distr1", "47_yeast_equal_distr1"], #["29_Pima1", "47_yeast1"]
                         ["AlbaMethod","SmartInitialGuess", "RandomSampling", "BasicActiveLearning"],
                         10)
    pp()
    #pp._fig()