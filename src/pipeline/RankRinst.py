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
    def __init__(self, datasets, steps) -> None:
        self.datasets = datasets
        self.steps = steps
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
            namesDatasets = {}
            for i in range(len(self.datasets)):
                namesDatasets[self.datasets[i]] = i
            #rankstdev = np.zeros((len(self.datasets)))
            for nn in range(len(self.datasets)):
                probs = np.zeros((101,500))
                rank = np.zeros((101,500))
                rankmean = np.zeros((500))
                filename = os.path.join("rints",self.datasets[nn]+".csv")
                with open(filename, 'r') as readFile:
                    reader = csv.reader(readFile)
                    lines = list(reader)
                    labels = lines[0]
                    labels = np.rint(np.array([float(i) for i in labels[1:]]))
                with open(os.path.join("probs", self.datasets[nn]+".SmartInitialGuess.7.csv"), 'r') as readFile:
                    reader = csv.reader(readFile)
                    lines = list(reader)
                    for line in range(len(lines)):
                        probs[line,:] = np.asarray([float(ww) for ww in (lines[line])[1:]], dtype = float)
    
                ## CALCULATE RANK
                rank[:,:] = rankdata(-probs, axis = 1,method='min') 
                if np.any(rank>500):
                    print(rank[rank>500])
                    raise Exception("")
                
                
                ## calculate mean
                rankmean[:] = np.mean(rank, axis = 0)

                idxs = labels == 1
                
                print(np.mean(rankmean[idxs]))
                return

                """## calculate variance
                rankstdev[:] = np.std(rank, axis = 0)
                print(rankmean)"""
            return

if __name__=="__main__":
    pp = PlotPerformance(["40_Vowels", "47_yeast","20_letter", "12_Fault",  "41_Waveform"],#["29_Pima_equal_distr1", "47_yeast_equal_distr1"], #["29_Pima1", "47_yeast1"]
                         10)
    pp()
    #pp._fig()