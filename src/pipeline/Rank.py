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

class PlotPerformance:
    def __init__(self, datasets, methods, count, steps) -> None:
        self.datasets = datasets
        self.methods = methods
        self.count = count
        self.steps = steps
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
            namesDatasets = {}
            for i in range(len(self.datasets)):
                namesDatasets[self.datasets[i]] = i
            namesMethods = {}
            for i in range(len(self.methods)):
                namesMethods[self.methods[i]] = i
            aucroc = np.zeros((len(self.methods),101,len(self.datasets), self.count))
            aucrocmean = np.zeros((len(self.methods),101,len(self.datasets)))
            aucrocstdev = np.zeros((len(self.methods),101,len(self.datasets)))
            aucrocbag = np.zeros((len(self.methods),101,len(self.datasets), self.count))
            aucrocbagmean = np.zeros((len(self.methods),101,len(self.datasets)))
            aucrocbagstdev = np.zeros((len(self.methods),101,len(self.datasets)))
            rank = np.zeros((len(self.methods),101,len(self.datasets), self.count))
            rankbag = np.zeros((len(self.methods),101,len(self.datasets), self.count))
            """rankmean = np.zeros((len(self.methods),101,len(self.datasets)))
            rankstdev = np.zeros((len(self.methods),101,len(self.datasets)))
            rankbag = np.zeros((len(self.methods),101,len(self.datasets), self.count))
            rankbagmean = np.zeros((len(self.methods),101,len(self.datasets)))
            rankbagstdev = np.zeros((len(self.methods),101,len(self.datasets)))"""
            rankmean = np.zeros((len(self.methods),101))
            rankstdev = np.zeros((len(self.methods),101))
            rankbagmean = np.zeros((len(self.methods),101))
            rankbagstdev = np.zeros((len(self.methods),101))
            lastmethod = ""
            iteration = 0
            for nn in range(len(self.datasets)):
                directory = os.path.join(current, "results",self.datasets[nn])
                import csv
                for filename in os.listdir(directory):
                    f = os.path.join(directory, filename)
                    # checking if it is a file
                    if os.path.isfile(f):
                        with open(f, 'r') as readFile:
                            reader = csv.reader(readFile)
                            lines = list(reader)
                            currentname = f.split("\\")[-1].split(".")[0]
                            print(currentname, iteration)
                            if currentname in self.methods:
                                if iteration < self.count:
                                    if lastmethod == "":
                                        lastmethod = currentname
                                    if lastmethod != "" and lastmethod != currentname:
                                        print(currentname, lastmethod, iteration, self.count)
                                        raise Exception("Implementation error")
                                    for line in lines:
                                        if line[0] == "auc_roc":
                                            aucroc[namesMethods[currentname],:, namesDatasets[self.datasets[nn]],iteration] = np.asarray([float(ww) for ww in line[1:]], dtype = float)
                                        if line[0] == "auc_roc_bag":
                                            aucrocbag[namesMethods[currentname],:, namesDatasets[self.datasets[nn]],iteration] = np.asarray([float(ww) for ww in line[1:]], dtype = float)
                                    iteration += 1
                                else:
                                    if currentname == lastmethod:
                                        print("CONTINUED")
                                        continue
                                    lastmethod = currentname
                                    iteration = 0
                                    for line in lines:
                                        if line[0] == "auc_roc":
                                            aucroc[namesMethods[currentname],:, namesDatasets[self.datasets[nn]],iteration] = np.asarray([float(ww) for ww in line[1:]], dtype = float)
                                        if line[0] == "auc_roc_bag":
                                            aucrocbag[namesMethods[currentname],:, namesDatasets[self.datasets[nn]],iteration] = np.asarray([float(ww) for ww in line[1:]], dtype = float)
                                    iteration += 1
            ## calculate rank
            rank[:,:,:,:] = rankdata(-aucroc, axis = 0,method='min') 
            for i in range(101):
                for j in range(len(self.datasets)):
                    for h in range(self.count):
                        if np.all(np.sort(rank[:,i,j,h])== np.array(len(self.methods))):
                            raise Exception("False rnak " +str(rank[:,i,j,h])+" "+str(aucroc[:,i,j,h]))
            rankbag[:,:,:,:] = rankdata(-aucrocbag, axis = 0,method='min') 
            for i in range(101):
                for j in range(len(self.datasets)):
                    for h in range(self.count):
                        if np.all(np.sort(rankbag[:,i,j,h])== np.array(len(self.methods))):
                            raise Exception("False rnak " +str(rankbag[:,i,j,h])+" "+str(aucrocbag[:,i,j,h]))
            
            
            ## calculate mean
            rankmean[:,:] = np.mean(rank, axis = (2,3))
            rankbagmean[:,:] = np.mean(rankbag, axis = (2,3))
            aucrocmean[:,:,:] = np.mean(aucroc, axis = -1)
            aucrocbagmean[:,:,:] = np.mean(aucrocbag, axis = -1)

            ## calculate variance
            rankstdev[:,:] = np.std(rank, axis = (2,3))
            rankbagstdev[:,:] = np.std(rankbag, axis = (2,3))
            aucrocstdev[:,:,:] = np.std(aucroc, axis = -1)
            aucrocbagstdev[:,:,:] = np.std(aucrocbag, axis = -1)
            
            totalSteps =1+ 101//self.steps
            if totalSteps == 102:
                totalSteps = 101
            www = []
            tt = np.arange(totalSteps)
            xx = self.steps * np.arange(totalSteps)
            for i in xx:
                www.append(i)

            print(np.argmax(aucroc[:,:,:,:], axis = 0).shape)
            print("INSTANCE LEVEL")
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucroc[:,:,:,:], axis = 0))
            print("Overall", dummy, np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape))
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucroc[:,:,:,:], axis = 0))
            dummy = np.sum(np.logical_or(1== rank[namesMethods["SmartInitialGuess"],:,:,:], 2== rank[namesMethods["SmartInitialGuess"],:,:,:]))
            print("In Top2", dummy, np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape))
            methods = [namesMethods["SmartInitialGuess"], namesMethods["AlbaMethod"]]
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucroc[methods,:,:,:], axis = 0))
            print("ALBA", dummy, np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape))
            methods = [namesMethods["SmartInitialGuess"], namesMethods["BasicActiveLearning"]]
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucroc[methods,:,:,:], axis = 0))
            print("US", dummy, np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape))
            methods = [namesMethods["SmartInitialGuess"], namesMethods["RandomSampling"]]
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucroc[methods,:,:,:], axis = 0))
            print("RS", dummy, np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(aucroc[namesMethods["SmartInitialGuess"],:,:,:].shape))
            dummy =  np.divide(aucroc[namesMethods["SmartInitialGuess"],0,:,:], aucroc[namesMethods["AlbaMethod"],0,:,:])
            print("ImprovementAligning", (np.mean(dummy) -1)*100)
            print("BAG LEVEL")
            
            """dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucrocbag[:,:,namesDatasets["29_Pima1"],:], axis = 0))
            print("Overall Pima", dummy, np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,namesDatasets["29_Pima1"],:].shape),dummy/ np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,namesDatasets["29_Pima1"],:].shape))
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucrocbag[:,:,namesDatasets["47_yeast1"],:], axis = 0))
            print("Overall Yeast", dummy, np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,namesDatasets["47_yeast1"],:].shape),dummy/ np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,namesDatasets["47_yeast1"],:].shape))
            """
            
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucrocbag[:,:,:,:], axis = 0))
            print("Overall", dummy, np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,:,:].shape))
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucrocbag[:,:,:,:], axis = 0))
            dummy = np.sum(np.logical_or(1== rankbag[namesMethods["SmartInitialGuess"],:,:,:], 2== rankbag[namesMethods["SmartInitialGuess"],:,:,:]))
            print("In Top2", dummy, np.prod(rankbag[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(rankbag[namesMethods["SmartInitialGuess"],:,:,:].shape))
            methods = [namesMethods["SmartInitialGuess"], namesMethods["AlbaMethod"]]
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucrocbag[methods,:,:,:], axis = 0))
            print("ALBA", dummy, np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,:,:].shape))
            methods = [namesMethods["SmartInitialGuess"], namesMethods["BasicActiveLearning"]]
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucrocbag[methods,:,:,:], axis = 0))
            print("US", dummy, np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,:,:].shape))
            methods = [namesMethods["SmartInitialGuess"], namesMethods["RandomSampling"]]
            dummy = np.sum(namesMethods["SmartInitialGuess"]== np.argmax(aucrocbag[methods,:,:,:], axis = 0))
            print("RS", dummy, np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,:,:].shape),dummy/ np.prod(aucrocbag[namesMethods["SmartInitialGuess"],:,:,:].shape))
            idxs = aucrocbag[namesMethods["AlbaMethod"],0,:,:] != 0
            dummy =  np.divide(aucrocbag[namesMethods["SmartInitialGuess"],0,:,:][idxs], aucrocbag[namesMethods["AlbaMethod"],0,:,:][idxs])
            print("ImprovementAligning", (np.mean(dummy) -1)*100)
            

            rankmean = np.around(rankmean, decimals=2)
            rankstdev = np.around(rankstdev, decimals=2)
            rankbagmean = np.around(rankbagmean, decimals=2)
            rankbagstdev = np.around(rankbagstdev, decimals=2)
            aucrocbagmean = np.around(aucrocbagmean, decimals=2)
            aucrocmean = np.around(aucrocmean, decimals=2)
            aucrocstdev = np.around(aucrocstdev, decimals=2)
            aucrocbagstdev = np.around(aucrocbagstdev, decimals=2)

            with open(os.path.join("ranks","totalRanks.csv"), 'w', newline="") as file:
                csvwriter = csv.writer(file) # 2. create a csvwriter object
                csvwriter.writerow(["method"] + list(www)) # 4. write the header
                for i in range(len(self.methods)):
                    row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b) for a, b in zip(rankmean[i, xx], rankstdev[i, xx])]
                    csvwriter.writerow([self.methods[i]] + row)

            """for dataset in namesDatasets.keys():
                with open(os.path.join("ranks",dataset+".csv"), 'w', newline="") as file:
                    csvwriter = csv.writer(file) # 2. create a csvwriter object
                    csvwriter.writerow(["method"] + list(www)) # 4. write the header
                    for i in range(len(self.methods)):
                        row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b) for a, b in zip(rankmean[i, xx, namesDatasets[dataset]], rankstdev[i, xx, namesDatasets[dataset]])]
                        csvwriter.writerow([self.methods[i]] + row)"""

            with open(os.path.join("ranksbag","totalRanksBag.csv"), 'w', newline="") as file:
                csvwriter = csv.writer(file) # 2. create a csvwriter object
                csvwriter.writerow(["method"] + list(www)) # 4. write the header
                for i in range(len(self.methods)):
                    row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b) for a, b in zip(rankbagmean[i, xx], rankbagstdev[i, xx])]
                    csvwriter.writerow([self.methods[i]] + row)
            
            """for dataset in namesDatasets.keys():
                with open(os.path.join("ranksbag",dataset+".csv"), 'w', newline="") as file:
                    csvwriter = csv.writer(file) # 2. create a csvwriter object
                    csvwriter.writerow(["method"] + list(www)) # 4. write the header
                    for i in range(len(self.methods)):
                        row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b) for a, b in zip(rankbagmean[i, xx, namesDatasets[dataset]], rankbagstdev[i, xx, namesDatasets[dataset]])]
                        csvwriter.writerow([self.methods[i]] + row)"""
                        
            for dataset in namesDatasets.keys():            
                with open(os.path.join("instancelevel",dataset+".csv"), 'w', newline="") as file:
                    csvwriter = csv.writer(file) # 2. create a csvwriter object
                    csvwriter.writerow(["method"] + list(www)) # 4. write the header
                    for i in range(len(self.methods)):
                        row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b) for a, b in zip(aucrocmean[i, xx, namesDatasets[dataset]], aucrocstdev[i, xx, namesDatasets[dataset]])]
                        csvwriter.writerow([self.methods[i]] + row)
                    """for i in range(len(self.methods)):
                        row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b) for a, b in zip(rankmean[i, xx, namesDatasets[dataset]], rankstdev[i, xx, namesDatasets[dataset]])]
                        csvwriter.writerow([self.methods[i]] + row)"""
                        
                with open(os.path.join("baglevel",dataset+".csv"), 'w', newline="") as file:
                    csvwriter = csv.writer(file) # 2. create a csvwriter object
                    csvwriter.writerow(["method"] + list(www)) # 4. write the header
                    for i in range(len(self.methods)):
                        row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b) for a, b in zip(aucrocbagmean[i, xx, namesDatasets[dataset]], aucrocbagstdev[i, xx, namesDatasets[dataset]])]
                        csvwriter.writerow([self.methods[i]] + row)
                    """for i in range(len(self.methods)):
                        row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b) for a, b in zip(rankbagmean[i, xx, namesDatasets[dataset]], rankbagstdev[i, xx, namesDatasets[dataset]])]
                        csvwriter.writerow([self.methods[i]] + row)"""

                        
                        
                with open(os.path.join("aucroc",dataset+".csv"), 'w', newline="") as file:
                    csvwriter = csv.writer(file) # 2. create a csvwriter object
                    csvwriter.writerow(["method"] + list(www)) # 4. write the header
                    for i in range(len(self.methods)):
                        row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b) for a, b in zip(aucrocmean[i, xx, namesDatasets[dataset]], aucrocstdev[i, xx, namesDatasets[dataset]])]
                        csvwriter.writerow([self.methods[i]] + row)
                        
                        
                with open(os.path.join("aucrocbag",dataset+".csv"), 'w', newline="") as file:
                    csvwriter = csv.writer(file) # 2. create a csvwriter object
                    csvwriter.writerow(["method"] + list(www)) # 4. write the header
                    for i in range(len(self.methods)):
                        row = ["{:.2f}".format(a)+ " + " + "{:.2f}".format(b)  for a, b in zip(aucrocbagmean[i, xx, namesDatasets[dataset]], aucrocbagstdev[i, xx, namesDatasets[dataset]])]
                        csvwriter.writerow([self.methods[i]] + row)
            return

    def _fig(self, *args: Any, **kwds: Any) -> Any:
                    
            fig, ax = plt.subplots( nrows=1, ncols=1, figsize = (10,6) ) 
            import csv
            with open(os.path.join(current,'auc_roc.csv'), 'r') as readFile:
                reader = csv.reader(readFile)
                lines = list(reader)


            dummy = np.Inf
            dummyName = ""
            dummyList = None

            mmm = "AAA"


            for line in lines:
                if int(line[0].split(".")[-1]) == dummy:
                    dummy += 1
                    dummyList = dummyList + np.asarray([float(i) for i in line[1:]])
                else:
                    if dummyName != "" and (mmm in dummyName):
                        ax.plot(dummyList/dummy, label = dummyName)
                    dummy = 1
                    dummyList = np.asarray([float(i) for i in line[1:]])
                    dummyName = ".".join(line[0].split(".")[:-1])
                #if "random" in line[0] or True:
                #    ax.plot([float(i) for i in line[1:]], label = line[0])
            if dummyName != "" and (mmm in dummyName):
                ax.plot(dummyList/dummy, label = dummyName)
            plt.title('Wine', fontsize=20)
            plt.ylabel('ROC AUC',fontsize=18)
            plt.xlabel('Round i',fontsize=18)
            plt.legend(fontsize = 16)
            plt.ylim([0,1])
            plt.xticks(fontsize = 13)
            plt.yticks(fontsize = 13)
            fig.savefig(os.path.join(current,'img/ROC AUC.png'),bbox_inches='tight')
            plt.show()
            plt.close(fig)

    def plotTimes(self):
            import csv
            title = "Average running time per instance query"
            xlabel = "Dataset"
            ylabel = "Elapsed time (s)"
            savename = "times.png"

            fig, ax = plt.subplots(layout='constrained')

            with open(os.path.join(current,'times.csv'), 'r') as readFile:
                reader = csv.reader(readFile)
                lines = list(reader)

            times = {}
            methodnames = []
            xlabels = []

            for line in lines:
                name = line[0].split(".")
                if name[1] in times.keys():
                    times[name[1]].append(np.mean(np.asarray([float(i) for i in line[1:]])))
                else:
                    methodnames.append(name[1])
                    times[name[1]] = [np.mean(np.asarray([float(i) for i in line[1:]]))]
                if name[0] not in xlabels:
                    xlabels.append(name[0])

            xs =np.arange(len(xlabels))
            width = 1/(len(methodnames)+2)
            multiplier = 0

            # code inspired by https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
            for k, v in times.items():
                """
                ## To plot speedups:
                if k == "A Tree per Batch":
                    v0 = v.copy()
                else:
                    v1 = v.copy()
                    for i in range(len(v0)):
                        v1[i] = v1[i]/v0[i]
                ##
                """
                offset = width * multiplier
                rects = ax.bar(xs+offset, [i/4 for i in v], width, label = k)
                #ax.set_yscale('log')
                ax.bar_label(rects, padding=3)
                multiplier +=1

            #offset = width * multiplier
            #rects = ax.bar(xs+offset, v1, width, label = "division")

            #ax.set_yscale('log')
            ax.bar_label(rects, padding=3)

            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            ax.set_xticks(xs+((multiplier-1)/2)*width, xlabels)
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig(savename)
            plt.close()


    def plotTimes2(self):
            import csv
            title = "Time performance for different active learning strategies"
            xlabel = "Method"
            ylabel = "Average running time per instance query (s)"
            savename = "times.png"

            fig, ax = plt.subplots(layout='constrained')

            with open(os.path.join(current,'times.csv'), 'r') as readFile:
                reader = csv.reader(readFile)
                lines = list(reader)

            times = {}
            methodnames = []
            xlabels = []

            for line in lines:
                name = line[0].split(".")
                if name[1] in times.keys():
                    times[name[1]].append(np.mean(np.asarray([float(i) for i in line[1:]])))
                else:
                    methodnames.append(name[1])
                    times[name[1]] = [np.mean(np.asarray([float(i) for i in line[1:]]))]
                if name[0] not in xlabels:
                    xlabels.append(name[0])
            print(times)
            xs =np.arange(len(methodnames))
            width = .5
            multiplier = 0

            # code inspired by https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
            rects = ax.bar(xs, [i[0]/4 for i in times.values()], width)
            #ax.set_yscale('log')
            ax.bar_label(rects, padding=3)
            
            #offset = width * multiplier
            #rects = ax.bar(xs+offset, v1, width, label = "division")

            #ax.set_yscale('log')
            ax.bar_label(rects, padding=3)

            plt.title(title, fontsize = 17)
            plt.xlabel(xlabel, fontsize = 14)
            plt.ylabel(ylabel, fontsize = 14)
            plt.ylim([0,470])
            ax.set_xticks(xs, methodnames)
            plt.grid()
            plt.show()
            plt.savefig(savename)
            plt.close()

if __name__=="__main__":
    """pp = PlotPerformance(["40_Vowels0", "47_yeast0","20_letter0","12_Fault0",   "41_Waveform0"],#["29_Pima_equal_distr1", "47_yeast_equal_distr1"], #["29_Pima1", "47_yeast1"]
                         ["SmartInitialGuess",  "AlbaMethod","BasicActiveLearning", "RandomSampling"], 
                         2, 
                         20)"""
    pp = PlotPerformance(["40_vowels_equal_distr0", "47_yeast_equal_distr0", "20_Letter_equal_distr0"],#["29_Pima_equal_distr1", "47_yeast_equal_distr1"], #["29_Pima1", "47_yeast1"]
                         ["SmartInitialGuess",  "AlbaMethod","BasicActiveLearning", "RandomSampling"], 
                         2, 
                         20)
    pp()
    #pp._fig()