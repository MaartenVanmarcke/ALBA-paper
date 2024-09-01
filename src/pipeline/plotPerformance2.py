from typing import Any
import matplotlib.pyplot as plt
import os
import pathlib
import numpy as np
current = pathlib.Path().absolute()
current = os.path.join(current, "src")
import numpy as np
p =  os.path.join(current, "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)

class PlotPerformance:
    def __init__(self, name, title) -> None:
        self.name = name
        self.title = title
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
            if len(self.name)>3:
                fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize = (13,8))
                fig.subplots_adjust(hspace=0.01, wspace=0.01)
                axs[0][2].set_visible(False)
                xaxs = [0,0,1,1,1]
                yaxs = [0,1,0,1,2]
            else:
                fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize = (13,4.5))
                fig.subplots_adjust(hspace=0.01, wspace=0.01)
                axs = [axs]
                xaxs = [0,0,0]
                yaxs = [0,1,2]
            for nn in range(len(self.name)):
                names = []
                countsaucroc = []
                countsaucrocbag = []
                aucroc = []
                aucrocbag = []
                directory = os.path.join(current, "results",self.name[nn])
                import csv
                for filename in os.listdir(directory):
                    f = os.path.join(directory, filename)
                    # checking if it is a file
                    if os.path.isfile(f):
                        with open(f, 'r') as readFile:
                            reader = csv.reader(readFile)
                            lines = list(reader)
                            currentname = f.split("\\")[-1].split(".")[0]
                            if len(names)> 0 and names[-1] == currentname:
                                for line in lines:
                                    if line[0] == "auc_roc":
                                        aucroc[-1] = aucroc[-1] + np.asarray([float(ww) for ww in line[1:]], dtype = float)
                                        countsaucroc[-1] += 1
                                    if line[0] == "auc_roc_bag":
                                        aucrocbag[-1] = aucrocbag[-1] + np.asarray([float(ww) for ww in line[1:]], dtype = float)
                                        countsaucrocbag[-1] += 1
                            else:
                                names.append(f.split("\\")[-1].split(".")[0])
                                for line in lines:
                                    if line[0] == "auc_roc":
                                        aucroc.append( np.asarray([float(ww) for ww in line[1:]], dtype = float))
                                    if line[0] == "auc_roc_bag":
                                        aucrocbag.append( np.asarray([float(ww) for ww in line[1:]], dtype = float))
                                countsaucroc.append(1)
                                countsaucrocbag.append(1)
                todel = None
                toput = None
                for i in range(len(names)):
                    if names[i] == "AlbaMethod":
                        names[i] = "ALBA"
                    if names[i] == "SmartInitialGuess":
                        names[i] = "AMIB"
                        toput = i
                    if names[i] == "RandomSampling":
                        names[i] = "Random Sampling"
                    if names[i] == "BasicActiveLearning":
                        names[i] = "Uncertainty Sampling"
                    if names[i] == "WithoutAlignment":
                        todel= i
                if todel != None:
                    names.pop(todel)
                    countsaucroc.pop(todel)
                    countsaucrocbag.pop(todel)
                    aucroc.pop(todel)
                    aucrocbag.pop(todel)
                if toput > todel:
                    toput -= 1
                name = names.pop(toput)
                names.insert(0,name)
                score = countsaucroc.pop(toput)
                countsaucroc.insert(0,score)
                score = countsaucrocbag.pop(toput)
                countsaucrocbag.insert(0,score)
                score = aucroc.pop(toput)
                aucroc.insert(0,score)
                score = aucrocbag.pop(toput)
                aucrocbag.insert(0,score)

                if False:
                    names.append("SIGEnsemble")
                    countsaucrocbag.append(1)
                    aucrocbag.append(np.zeros((0)))
                    i = 0
                    while names[i] != "SmartInitialGuess":
                        i += 1 
                    dummy = aucroc[i]
                    fltr = np.ones((5))/5
                    dummy = np.convolve(dummy,fltr, mode="same")
                    print(dummy)
                    aucroc.append(dummy)
                    countsaucroc.append(countsaucroc[i])
                colors = ["#d62728", "#2ca02c", "#ff7f0e","#1f77b4"]
                ax = axs[xaxs[nn]][yaxs[nn]]
                for i in range(len(names)):
                    if names[i] == "AMIB":
                        ax.plot(np.arange(0,len(aucroc[i])),aucroc[i]/countsaucroc[i], label = names[i],linewidth=2.0, c= colors[i])
                    else:
                        ax.plot(np.arange(0,len(aucroc[i])),aucroc[i]/countsaucroc[i],"--", label = names[i], c= colors[i])
                    ax.set_ylim(0.3,1)
                    ax.set_xlim(0,100)
                    #ax.set_title(self.title[nn])
                    ax.set_xticks(np.arange(20,91,20))
                    if len(self.name)>3:
                        ax.set_yticks(np.arange(0.4,1,0.1))
                    ax.grid(True)
                    handles, labels = ax.get_legend_handles_labels()
                    # these are matplotlib.patch.Patch properties
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                    # place a text box in upper left in axes coords
                    ax.text(0.95, 0.05, self.title[nn], transform=ax.transAxes,
                        verticalalignment='bottom', horizontalalignment='right', bbox=props, fontsize = 18)
            fig.legend(handles, labels, loc = (0.12, .89), ncol = 5, fontsize = 17)#loc='upper center', ncol=5)
            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Nr. of instances labeled", fontsize = 17)
            plt.ylabel("AUROC", fontsize = 17)
            #plt.savefig("AUCROC.png",bbox_inches='tight')
            #plt.legend()
            plt.show()

            
            if len(self.name)>3:
                fig, axs = plt.subplots(2,3, sharex=True, sharey=True, figsize = (13,8))
                fig.subplots_adjust(hspace=0.01, wspace=0.01)
                axs[0][2].set_visible(False)
                xaxs = [0,0,1,1,1]
                yaxs = [0,1,0,1,2]
            else:
                fig, axs = plt.subplots(1,3, sharex=True, sharey=True, figsize = (13,4.5))
                fig.subplots_adjust(hspace=0.01, wspace=0.01)
                axs = [axs]
                xaxs = [0,0,0]
                yaxs = [0,1,2]
            for nn in range(len(self.name)):
                ax = axs[xaxs[nn]][yaxs[nn]]
                names = []
                countsaucroc = []
                countsaucrocbag = []
                aucroc = []
                aucrocbag = []
                directory = os.path.join(current, "results",self.name[nn])
                import csv
                minim = 0
                maxim = 1
                for filename in os.listdir(directory):
                    f = os.path.join(directory, filename)
                    # checking if it is a file
                    if os.path.isfile(f):
                        with open(f, 'r') as readFile:
                            reader = csv.reader(readFile)
                            lines = list(reader)
                            currentname = f.split("\\")[-1].split(".")[0]
                            if len(names)> 0 and names[-1] == currentname:
                                for line in lines:
                                    if line[0] == "auc_roc":
                                        aucroc[-1] = aucroc[-1] + np.asarray([float(ww) for ww in line[1:]], dtype = float)
                                        countsaucroc[-1] += 1
                                    if line[0] == "auc_roc_bag":
                                        aucrocbag[-1] = aucrocbag[-1] + np.asarray([float(ww) for ww in line[1:]], dtype = float)
                                        countsaucrocbag[-1] += 1
                            else:
                                names.append(f.split("\\")[-1].split(".")[0])
                                for line in lines:
                                    if line[0] == "auc_roc":
                                        aucroc.append( np.asarray([float(ww) for ww in line[1:]], dtype = float))
                                    if line[0] == "auc_roc_bag":
                                        aucrocbag.append( np.asarray([float(ww) for ww in line[1:]], dtype = float))
                                countsaucroc.append(1)
                                countsaucrocbag.append(1)
                todel = None
                toput = None
                for i in range(len(names)):
                    if names[i] == "AlbaMethod":
                        names[i] = "ALBA"
                    if names[i] == "SmartInitialGuess":
                        names[i] = "AMIB"
                        toput = i
                    if names[i] == "RandomSampling":
                        names[i] = "Random Sampling"
                    if names[i] == "BasicActiveLearning":
                        names[i] = "Uncertainty Sampling"
                    if names[i] == "WithoutAlignment":
                        todel= i
                if todel != None:
                    names.pop(todel)
                    countsaucroc.pop(todel)
                    countsaucrocbag.pop(todel)
                    aucroc.pop(todel)
                    aucrocbag.pop(todel)
                if toput > todel:
                    toput -= 1
                name = names.pop(toput)
                names.insert(0,name)
                score = countsaucroc.pop(toput)
                countsaucroc.insert(0,score)
                score = countsaucrocbag.pop(toput)
                countsaucrocbag.insert(0,score)
                score = aucroc.pop(toput)
                aucroc.insert(0,score)
                score = aucrocbag.pop(toput)
                aucrocbag.insert(0,score)

                if False:
                    names.append("SIGEnsemble")
                    countsaucrocbag.append(1)
                    aucrocbag.append(np.zeros((0)))
                    i = 0
                    while names[i] != "SmartInitialGuess":
                        i += 1 
                    dummy = aucroc[i]
                    fltr = np.ones((5))/5
                    dummy = np.convolve(dummy,fltr, mode="same")
                    print(dummy)
                    aucroc.append(dummy)
                    countsaucroc.append(countsaucroc[i])
                colors = ["#d62728", "#2ca02c", "#ff7f0e","#1f77b4"]
                for i in range(len(names)):
                    if names[i] == "AMIB":
                        ax.plot(np.arange(0,len(aucrocbag[i])),aucrocbag[i]/countsaucrocbag[i], label = names[i],linewidth=2.0, c= colors[i])
                    else:
                        ax.plot(np.arange(0,len(aucrocbag[i])),aucrocbag[i]/countsaucrocbag[i],"--", label = names[i], c= colors[i])
                    ax.set_ylim(0.3,1)
                    ax.set_xlim(0,100)
                    #ax.set_title(self.title[nn])
                    ax.set_xticks(np.arange(20,91,20))
                    if len(self.name)>3:
                        ax.set_yticks(np.arange(0.4,1,0.1))
                    ax.grid(True)
                    handles, labels = ax.get_legend_handles_labels()
                    # these are matplotlib.patch.Patch properties
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                    # place a text box in upper left in axes coords
                    ax.text(0.95, 0.05, self.title[nn], transform=ax.transAxes,
                        verticalalignment='bottom', horizontalalignment='right', bbox=props, fontsize = 18)
            fig.legend(handles, labels, loc = (0.12, .89), ncol = 5, fontsize = 17)#loc='upper center', ncol=5)
            fig.add_subplot(111, frameon=False)
            # hide tick and tick label of the big axis
            plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Nr. of instances labeled", fontsize = 17)
            plt.ylabel("AUROC", fontsize = 17)
            #plt.savefig("AUCROC.png",bbox_inches='tight')
            #plt.legend()
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

            plt.title(title, fontsize = 20)
            plt.xlabel(xlabel, fontsize = 16)
            plt.ylabel(ylabel, fontsize = 16)
            plt.ylim([0,470])
            ax.set_xticks(xs, methodnames)
            plt.grid()
            plt.show()
            plt.savefig(savename)
            plt.close()

if __name__=="__main__":
    #pp = PlotPerformance(["47_yeast0", "47_yeast0","47_yeast0","47_yeast0",   "47_yeast0"],["Vowels", "Yeast", "Letter","Fault", "Waveform"])
    pp = PlotPerformance(["40_vowels_equal_distrNew0","47_yeast_equal_distrNew0","20_letter_equal_distrNew0"],["Vowels", "Yeast", "Letter"])
    pp = PlotPerformance(["40_vowels_equal_distrNew0","47_yeast_equal_distrNew0","20_letter_equal_distrNew0"],["Vowels", "Yeast", "Letter"])
    pp = PlotPerformance(["40_VowelsNew0","47_yeastNew0","20_letterNew0", '12_FaultNew0', '41_WaveformNew0'],["Vowels", "Yeast", "Letter","Fault", "Waveform"])
    pp()
    #pp._fig()