from typing import Any
import numpy as np
from pyod.models.iforest import IForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

class Checker:
    def __init__(self) -> None:
        pass

    def __call__(self, bags, bags_labels, X_inst, y_inst, *args: Any, **kwds: Any) -> Any:
        f1 = self.check1(bags, bags_labels, X_inst, y_inst)
        f2 = self.check2(bags, bags_labels, X_inst, y_inst)
        f3 = self.check3(bags, bags_labels, X_inst, y_inst)
        f4 = self.check4(bags, bags_labels, X_inst, y_inst)
        print(np.all(np.array([f1,f2,f3,f4])))
        if not np.all(np.array([f1,f2,f3,f4])):
            print("a")
            return False
            """print("Not all checks are satisfied: {}".format([f1,f2,f3,f4]))
            return True"""
            #raise Exception("Not all checks are satisfied: {}".format([f1,f2,f3,f4]))
        print("b")
        return True
        
    def check1(self, bags, bags_labels, X_inst, y_inst):
        valid = len(X_inst) == len(y_inst)
        valid = valid & (np.sum(np.bitwise_and(y_inst!=0,y_inst != 1))==0)
        if valid:
            print("CHECKED: All instance labels are available.")
            return True
        else:
            print("UNSATISFIED: Some instance labels are not available:\n{}".format(y_inst))
            return False
            raise Exception("Some instance labels are not available:\n{}".format(y_inst))

    def check2(self, bags, bags_labels, X_inst, y_inst):
        iforest = IForest(random_state=1302)
        iforest.fit(X_inst)
        prs = iforest.predict_proba(X_inst)[:,1]
        score = roc_auc_score(np.rint(y_inst), prs)
        if score<=.6:
            print("CHECKED: Anomalies are isolated. IF has an accuracy of {} <= 60%.".format(score))
            return True
        else:
            print("UNSATISFIED: Anomalies are not isolated.\nIF has an accuracy of {} > 60%.".format(score))
            return False
            raise Exception("Anomalies are not isolated.\nIF has an accuracy of {} > 60%.".format(score))
    
    def check3(self, bags, bags_labels, X_inst, y_inst):
        labels = {}
        idx = 0
        for key, bag in bags.items():
            l = bag.shape[0]
            labels[key] = y_inst[idx:idx+l]
            idx += l

        for bag1, data1 in bags.items():
            if bags_labels[bag1]>0:
                flag = False
                iforest = IForest(random_state=1302)
                iforest.fit(data1)
                prs = iforest.predict_proba(data1)[:,1]
                scoreOrig = roc_auc_score(np.rint(labels[bag1]), prs)
                for bag2, data2 in bags.items():
                    prs = iforest.predict_proba(data2)[:,1]
                    prs = np.concatenate((prs, np.array([1])))
                    dummy = np.concatenate((labels[bag2], np.array([1])))
                    score = roc_auc_score(np.rint(dummy), prs)
                    if abs(score-scoreOrig)>=.2:
                        flag = True
                        break
                if flag: 
                    continue
                else:
                    print("UNSATISFIED: The bags do not follow different distributions.")
                    return False
                    raise Exception("The bags do not follow different distributions.")
        print("CHECKED: The bags follow different distributions.")
        return True
    
    def check4(self, bags, bags_labels, X_inst, y_inst):
        bagIdx = np.zeros((0))
        for key, bag in bags.items():
            l = bag.shape[0]
            bagIdx = np.concatenate((bagIdx, key*np.ones((l))))
        anIdxs = y_inst > 0
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X_inst)
        _, indices = nbrs.kneighbors(X_inst[anIdxs,:])
        idxs = (bagIdx[anIdxs] != bagIdx[indices.T[1]])
        if np.sum(idxs) >= .6*len(idxs):
            print("CHECKED: Anomalies overlap with normals. {} >= .60%. of the anomalies have an NN of a different bag.".format(
                np.sum(idxs)/len(idxs)*100
            ))
            return True
        else:
            print("UNSATISFIED: Anomalies do not overlap with normals. Only {} < .60%. of the anomalies have an NN of a different bag. {} out of {} anomalies".format(
                np.sum(idxs)/len(idxs)*100, np.sum(idxs), len(idxs)
            ))
            return False
            raise Exception("Anomalies do not overlap with normals. Only {} < .60%. of the anomalies have an NN of a different bag.".format(
                np.sum(idxs)/len(idxs)*100
            ))
