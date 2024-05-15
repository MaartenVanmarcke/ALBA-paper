from typing import Any
import os
import csv
from datetime import date

class Saver():
    def __init__(self) -> None:
        self.globalParams = []
        pass

    def addGlobalParam(self, name, value):
        self.globalParams.append([name, value])

    def __call__(self, rewardInfo, current, dataName, methodName,time,parameters = None,*args: Any, **kwds: Any) -> Any:
        if parameters == None:
            parameters = []
        scoresAucRoc = ["auc_roc"] + rewardInfo.getAUC()["roc"]
        scoresAucPR = ["auc_pr"] + rewardInfo.getAUC()["pr"]
        scoresAucRocBag = ["auc_roc_bag"] + rewardInfo.getAUC()["rocbag"]
        scoresAucPRBag = ["auc_pr_bag"] + rewardInfo.getAUC()["prbag"]
        origdir = os.path.join(current, "results", dataName)
        mm = 0
        dir = origdir + str(mm)
        while os.path.exists(dir) & os.path.exists(os.path.join(dir, methodName+".csv")):
            mm += 1
            dir += str(mm)
        os.mkdir(dir)
        with open(os.path.join(dir, methodName+".csv"), 'w',newline='') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows([scoresAucRoc,
                              scoresAucPR,
                              scoresAucRocBag,
                              scoresAucPRBag,
                              ["Date", date.today()],
                              ["Time", time]] + self.globalParams + parameters)
        return None