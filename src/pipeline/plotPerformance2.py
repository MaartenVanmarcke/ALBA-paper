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
    def __init__(self, name) -> None:
        self.name = name
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
            names = []
            aucroc = []
            aucrocbag = []
            directory = os.path.join(current, "results",self.name)
            lastname = ""
            import csv
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                # checking if it is a file
                if os.path.isfile(f):
                    with open(f, 'r') as readFile:
                        reader = csv.reader(readFile)
                        lines = list(reader)
                        if names[-1] == lastname:
                            for line in lines:
                                if line[0] == "auc_roc":
                                    aucroc[-1] = aucroc[-1] + line[1:]
                                if line[0] == "auc_roc_bag":
                                    aucrocbag[-1] = aucrocbag[-1] + line[1:]
                        else:
                            names.append(f.split("\\")[-1].split(".")[0])
                            for line in lines:
                                if line[0] == "auc_roc":
                                    aucroc.append(line[1:])
                                if line[0] == "auc_roc_bag":
                                    aucrocbag.append(line[1:])
                            lastname = names[-1]
            plt.figure()
            for i in range(len(names)):
                plt.plot(np.arange(0,len(aucroc[i])),[float(k) for k in aucroc[i]], label = names[i])
            plt.ylim([0,1])
            plt.legend()
            plt.title(self.name)
            plt.ylabel("AUC ROC")
            plt.xlabel("Instances Queried")
            plt.grid()
            plt.show()
            plt.figure()
            for i in range(len(names)):
                plt.plot(np.arange(0,len(aucrocbag[i])),[float(k) for k in aucrocbag[i]], label = names[i])
            plt.ylim([0,1])
            plt.legend()
            plt.title(self.name)
            plt.ylabel("AUC ROC BAG")
            plt.xlabel("Instances Queried")
            plt.grid()
            plt.show()
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
    pp = PlotPerformance("2_annthyroid0")
    pp()
    #pp._fig()