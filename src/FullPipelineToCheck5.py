from typing import Any
from pipeline.constructBags import ConstructBags,ConstructBagsOpt
import pipeline.inputData as id
from pipeline.methods import WithAlignmentMethod, WithoutAlignmentMethod, Method, ActiveLearning, RandomSampling, SmartInitialGuessMethod, AlbaMethod
from pipeline.preprocessor import Preprocessor
from pipeline.plotPerformance import PlotPerformance
from pipeline.checker import Checker
from pipeline.saver import Saver
import time
import csv
import sys
import warnings
sys.path.insert(1, '../')
import os
import pathlib
current = pathlib.Path().parent.absolute()
p =  os.path.join(current, "src")
sys.path.insert(1, os.path.join(current))
p =  os.path.join(p, "seed.txt")
file = open(p)
seed = int(file.read())
file.close()

class FullPipeline:
    def __init__(self) -> None:
        pass

    def __call__(self, 
                 inputDatas,#: list[InputData],
                 constructBags,#: ConstructBags,
                 preprocessor,#: Preprocessor,
                 methods,#: list[Method],
                 plotPerformance,#: PlotPerformance,
                 checker,#: Checker,
                 saver,#:Saver,
                 *args: Any, **kwds: Any) -> Any:
        for inputData in inputDatas:
            normals = inputData.getNormals()
            anomalies = inputData.getAnomalies()
            seedcounter = 0
            query_budget = 100
            passsssed = []
            checkssss = []
            check4 = 0
            check4444 = []
            for i in ([12,14,19,23]):
                checkTime = 0
                nr_passed = 0
                seedcounter =i
                """file = open(p, mode = "w")
                file.write(str(i))
                file.close() """  
                bags, bags_labels, y_inst = constructBags.createBags(normals, anomalies, seedcounter)
                bags, bags_labels, X_inst, y_inst = preprocessor.standardize(bags, bags_labels, y_inst)
                flag = checker(bags, bags_labels, X_inst, y_inst)
                checkTime += 1
                if flag: 
                    nr_passed += 1
                print("CHECKTIME:", checkTime)
                print("NR Passed:", nr_passed)
            return 
            saver.addGlobalParam("seed", seedcounter)
            for method in methods:
                    start = time.time()
                    rewardInfo, current = method(inputData.getName(),query_budget, i, bags, bags_labels, X_inst, y_inst)
                    end = time.time()
                    self.savetime(inputData.getName() + "."+ method.name, i, end-start)
                    saver(rewardInfo, current, inputData.getName(), method.name+"."+str(i), end-start, parameters= None)
            seedcounter += 1
        plotPerformance()
        print("Pipeline finished without noticeable errors. :D")
        return None
    
    def savetime(self, name, n_experiment, time):
        with open(os.path.join(current, "src", 'times.csv'), 'r') as readFile:
            reader = csv.reader(readFile)
            lines = list(reader)
            flag = True
            for i in range(len(lines)):
                if lines[i][0] == name:
                    if n_experiment == 0:
                        lines[i] = [name, time]
                    else:
                        lines[i].append(time)
                    flag = False
            if flag:
                lines.append([name, time])
                        
        with open(os.path.join(current, "src", 'times.csv'), 'w',newline='') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lines)

if __name__=="__main__":
    inputDatas = [id.Fault_12()]#[TestData()]
    nclusters = 10
    instance_per_bag = 50
    nbags = 10
    bag_contfactor = .3
    constructBags = ConstructBagsOpt(nclusters,nbags, instance_per_bag,bag_contfactor)
    preprocessor = Preprocessor()
    checker = Checker()
    saver = Saver()
    saver.addGlobalParam("n_clusters", nclusters)
    saver.addGlobalParam("instance_per_bag", instance_per_bag)
    saver.addGlobalParam("n_bags", nbags)
    saver.addGlobalParam("bag_contfactor", bag_contfactor)
    """inputDatas = [TestData()]
    instance_per_bag = 8
    nbags = 5
    constructBags = ConstructBags(4,nbags, instance_per_bag,.3)
    preprocessor = Preprocessor()
    query_budget = int(.25*2)
    query_budget = 5"""
    methods = [AlbaMethod(),SmartInitialGuessMethod(),  WithoutAlignmentMethod(),ActiveLearning(),RandomSampling()]#,SmartInitialGuessMethod(),WithoutAlignmentMethod(),ActiveLearning(),RandomSampling()]#,WithAlignmentMethod() SmartInitialGuessMethod(), AlbaMethod(), ActiveLearning(), RandomSampling()]
    pipeline = FullPipeline()
    plotPerformance = PlotPerformance()
    pipeline(inputDatas, constructBags, preprocessor, methods, plotPerformance, checker, saver)