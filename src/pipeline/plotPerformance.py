from typing import Any
import matplotlib.pyplot as plt
import os
import pathlib
current = pathlib.Path().absolute()
current = os.path.join(current, "src")
import numpy as np
p =  os.path.join(current, "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)

class PlotPerformance:
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
                    
            fig, ax = plt.subplots( nrows=1, ncols=1)#, figsize = (16,9) ) 
            import csv
            with open(os.path.join(current,'auc_roc.csv'), 'r') as readFile:
                reader = csv.reader(readFile)
                lines = list(reader)

            lines = sorted(lines, key = (lambda x: x[0]))

            dummy = np.Inf
            dummyName = ""
            dummyList = None
            for line in lines:
                if int(line[0].split(".")[-1]) == dummy:
                    dummy += 1
                    dummyList = dummyList + np.asarray([float(i) for i in line[1:]])
                else:
                    if dummyName != "":
                        ax.plot(dummyList/dummy, label = dummyName)
                    dummy = 1
                    dummyList = np.asarray([float(i) for i in line[1:]])
                    dummyName = ".".join(line[0].split(".")[:-1])
                #if "random" in line[0] or True:
                #    ax.plot([float(i) for i in line[1:]], label = line[0])
            ax.plot(dummyList/dummy, label = dummyName)
            plt.title('ROC AUC after each round', fontsize=20)
            plt.ylabel('ROC AUC',fontsize=18)
            plt.xlabel('Round i',fontsize=18)
            plt.legend(fontsize = 16)
            plt.ylim([0,1])
            plt.xticks(fontsize = 13)
            plt.yticks(fontsize = 13)
            fig.savefig(os.path.join(current,'img/ROC AUC.png'),bbox_inches='tight')
            plt.show()
            plt.close(fig)

            fig, ax = plt.subplots( nrows=1, ncols=1, figsize = (16,9) ) 
            import csv
            with open(os.path.join(current,'auc_pr.csv'), 'r') as readFile:
                reader = csv.reader(readFile)
                lines = list(reader)

            for line in lines:
                if "random" in line[0] or True:
                    ax.plot([float(i) for i in line[1:]], label = line[0])
            plt.title('PR AUC')
            plt.ylabel('PR AUC')
            plt.xlabel('Round i')
            fig.legend()
            #plt.show()
            fig.savefig(os.path.join(current,'img/PR AUC.png'),bbox_inches='tight')
            plt.close(fig)


            ## BAGLEVEL

            fig, ax = plt.subplots( nrows=1, ncols=1, figsize = (16,9) ) 
            import csv
            with open(os.path.join(current,'auc_roc_bag.csv'), 'r') as readFile:
                reader = csv.reader(readFile)
                lines = list(reader)

            for line in lines:
                if "random" in line[0] or True:
                    ax.plot([float(i) for i in line[1:]], label = line[0])
            plt.title('ROC AUC BAG-LEVEL')
            plt.ylabel('ROC AUC')
            plt.xlabel('Round i')
            fig.savefig(os.path.join(current,'img/ROC AUC BAG.png'),bbox_inches='tight')
            fig.legend()
            #plt.show()
            plt.close(fig)

            fig, ax = plt.subplots( nrows=1, ncols=1, figsize = (16,9) ) 
            import csv
            with open(os.path.join(current,'auc_pr_bag.csv'), 'r') as readFile:
                reader = csv.reader(readFile)
                lines = list(reader)
                    
            for line in lines:
                if "random" in line[0] or True:
                    ax.plot([float(i) for i in line[1:]], label = line[0])
            plt.title('PR AUC BAG-LEVEL')
            plt.ylabel('PR AUC')
            plt.xlabel('Round i')
            fig.savefig(os.path.join(current,'img/PR AUC BAG.png'),bbox_inches='tight')
            fig.legend()
            #plt.show()
            plt.close(fig)

            ## TIMES

            self.plotTimes2()

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
    pp = PlotPerformance()
    pp._fig()