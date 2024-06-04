from typing import Any
import os
import pathlib
from alignflow_master.dataReplacer import DataReplacer
import numpy as np
current = pathlib.Path().absolute()
current = os.path.join(current, "src")
p =  os.path.join(current, "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)
import matplotlib
import matplotlib.pyplot as plt
import csv

class MainMethodFlattened():
    def __init__(self,
                versionname = "ds_4 normprobs threshold",
                original = False,
                al_strategy = "entropy",
                mabreward = None,
                load = False,

                query_budget = 1, #61 # nbs*k# 30# 30*10 # TODO
                restart = True,   # Do you restart the normalizing flows or do you finetune?
                smartguess = False,
                probs = None) -> None:
        
        self.mabreward = mabreward
        self.versionname = versionname
        self.original = original
        self.al_strategy = al_strategy
        self.load = load
        self.probs = probs

        self.query_budget = query_budget #61 # nbs*k# 30# 30*10 # TODO
        self.restart = restart   # Do you restart the normalizing flows or do you finetune?
        self.smartguess = smartguess
        pass

    def __call__(self, dataname, experimentnumber, bags, bags_labels, X_inst, y_inst, *args: Any, **kwds: Any) -> Any:
        versionname = dataname + "."+ self.versionname + "." + str(experimentnumber)
        original = self.original
        al_strategy = self.al_strategy
        load = self.load

        k = int(len(y_inst/len(bags)))          # number of instances/bag
        nbags = len(bags)     # number of bags
        query_budget = self.query_budget #61 # nbs*k# 30# 30*10 # TODO
        restart = self.restart   # Do you restart the normalizing flows or do you finetune?
        smartguess = self.smartguess

        # # The Main Loop
        # This does the full cycle for active learning and aligning bags.

        # ## Imports
        from data import Data
        from methods import MABMethod
        from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
        from dataBag import DataBag
        from rewardInfo import RewardInfo
        from itertools import cycle
        from alignflow_master.train_copyFullDataset import main
        # gen_data gives 3 dimensions.


        originalBags = bags.copy()
        D = bags
        bags = D

        # ## Set up variables
        originalBags = originalBags # The original bags, in dictionary format
        flattenedBags = {0:X_inst}
        flattenedBagLabel = np.asarray([np.any(bags_labels)])
        bags = bags # The original bags, in list of lists format
        dataBag = DataBag(bags, bags_labels, X_inst, y_inst) # The object that stores information about the labels and bag tensors.
        rewardInfo = RewardInfo(1) # The object that stores information about the process, for plotting purposes.
        dataReplacer = DataReplacer(nbags).setInitData(originalBags) # The object that sets the input for the normalizing flows
        alba = MABMethod(mab="rotting-swa", mabreward= self.mabreward,query_budget=query_budget, verbose=True, rewardInfo=rewardInfo , al_strategy = al_strategy) # The alba method
        training_data = Data(1).set_domains_and_labels(flattenedBags)

        # ## Some helper functions (TODO: check)
        def getPrediction(alba: MABMethod, data: Data):
            probabilities = alba.predict(data, self.probs, True)
            return probabilities

        def noisyOr(labels):
            lst = np.zeros((len(labels.keys())))
            for bag in labels.keys():
                lbls = np.array(labels[bag])
                lbls = 1-lbls
                lst[bag] = 1-np.prod(lbls)
            return lst

        def diff2dd(A, B):
            # https://stackoverflow.com/questions/8317022/get-intersecting-rows-across-two-2d-numpy-arrays
            nrows, ncols = A.shape
            dtype={'names':['f{}'.format(i) for i in range(ncols)],
                'formats':ncols * [A.dtype]}
            C = np.setdiff1d(A.view(dtype), B.view(dtype))
            # This last bit is optional if you're okay with "C" being a structured array...
            C = C.view(A.dtype).reshape(-1, ncols)
            return C

        def currentPrediction(alba: MABMethod, data: Data, bags: dict, newinstanceKey: int, instance, labeledPos = None, labeledNeg = None, t=0, weights = None):
            """
            t: iteration """
            if weights == None:
                weights = {}
                k = 0
                for bag in bags.keys():
                    weights[k] = np.ones((bags[bag].shape[0]))
                    k +=1

            probabilities = getPrediction(alba, data)
            prs = probabilities[0]

            probabilities = {}
            l = 0
            for bag in range(len(bags)):
                probabilities[bag] = prs[l:(l+len(bags[bag]))]
                l += len(bags[bag])


            scoreroc = roc_auc_score(np.rint(y_inst), prs)
            # Data to plot precision - recall curve
            precision, recall, _ = precision_recall_curve(np.rint(y_inst), prs)
            # Use AUC function to calculate the area under the curve of precision recall curve
            scorepr = auc(recall, precision)
            bagprobs = noisyOr(probabilities)
            scorerocBag = roc_auc_score(np.rint(bags_labels), bagprobs)
            # Data to plot precision - recall curve
            precision, recall, _ = precision_recall_curve(np.rint(bags_labels), bagprobs)
            # Use AUC function to calculate the area under the curve of precision recall curve
            scoreprBag = auc(recall, precision)
            rewardInfo.updateAuc("roc", scoreroc)
            rewardInfo.updateAuc("pr", scorepr)
            rewardInfo.updateAuc("rocbag", scorerocBag)
            rewardInfo.updateAuc("prbag", scoreprBag)
            
            with open(os.path.join("probs", str(versionname)+".csv"), 'a') as f_object:
 
                # Pass this file object to csv.writer()
                # and get a writer object
                writer_object = csv.writer(f_object)
            
                # Pass the list as an argument into
                # the writerow()
                writer_object.writerow([t] + list(prs))
            
                # Close the file object
                f_object.close()

            with open(os.path.join("bagprobs", str(versionname)+".csv"), 'a') as f_object:
 
                # Pass this file object to csv.writer()
                # and get a writer object
                writer_object = csv.writer(f_object)
            
                # Pass the list as an argument into
                # the writerow()
                writer_object.writerow([t] + list(bagprobs))
            
                # Close the file object
                f_object.close()


            if bags[0].shape[1] == 2:
                self.pplot(bags,probabilities, weights, instance, dataBag, t, scoreroc, scorepr, scorerocBag, scoreprBag)

            return probabilities


        def edgen(probs):
            return probs
            N = 20

            idxs = probs>.5
            print(idxs)
            print(np.invert(idxs))
            print(idxs.shape)
            a = np.invert(idxs)
            probs[a] = np.power(2*probs[a],2*N)/2
            probs[idxs] = 1-np.power(2*(probs[idxs]-1),2*N)/2
            return probs


        def predictBegin(test_data, probs):
                import transfer_learning_seperateArms as tl
                clsf = tl.get_transfer_classifier(
                        "none", "anomaly"
                    )  
                clsf.apply_transfer(test_data)
                clsf.fit_all(test_data,probs, ignore_unchanged=False)
                probs = clsf.predict(test_data, True, prior = probs)

                
                minim, maxim = np.Inf, -np.Inf
                for k,v in probs.items():
                    m = np.min(v) 
                    if m < minim:
                        minim = m
                    m = np.max(v)
                    if m > maxim:
                        maxim = m
                        
                predictions = {}
                allprobs = np.zeros((0))
                for k,v in probs.items():
                    predictions[k] = (v-minim)/(maxim-minim)
                    allprobs = np.concatenate((allprobs, predictions[k]))

                CONT_FACTOR = .1
                LENGTH = np.shape(allprobs)[0]
                thr_idx = LENGTH - int(np.ceil(LENGTH*CONT_FACTOR))
                idxs = np.argsort(allprobs)
                threshold = np.sum(allprobs[idxs[thr_idx-1:thr_idx+1]])/2
                
                summm = 0
                newpredictions = {}
                for k,v in predictions.items():
                    probabs = v/threshold
                    newpredictions[k] = 1-np.power(2, -probabs)
                    summm += np.sum(newpredictions[k]>=.5)

                return newpredictions

        if (original):
            t = 0  
            labeledPos = {}
            labeledNeg = {}
            for bag in range(len(bags)):
                labeledNeg[bag] = []
                labeledPos[bag] = []
            performance = []

            import warnings
            warnings.filterwarnings('ignore')

            for t in range(query_budget+1):
                queries = alba.fit_query(training_data, self.probs, True)

                #!!!!!!!
                key,idx = queries[0]
                i = 0
                while dataBag.isLabeled(key,idx):
                    i+=1
                    key, idx = queries[i]
                rewardInfo.chooseArm(key)
                key,idx = queries[i]
                instance = training_data.get_domain(key)[idx]
                lbl = dataBag.getLabel(key, idx)
                training_data.set_new_label(key, idx, lbl)
                training_data.set_last_labeled(key,idx)
                dataBag.label(key,idx)

                bb,ii = dataBag.findBagIdx(dataBag.findFullIdx(key,idx))
                rewardInfo.chooseArm2(bb)

                if (lbl == 1):
                    labeledPos[key].append(instance.tolist())
                else:
                    labeledNeg[key].append(instance.tolist())

                currentPrediction(alba, training_data, bags, key, instance, labeledPos,labeledNeg, t)
                
                #performance.append(dataBag.measureAccuracy(predictions))
                    
                print("=================")
                print("ITERATION", t)
                print("=================")

            ## window = 10!!!

        # %%
        if (not original):
            raise NotImplementedError("ImplementationError: This part should not be called.")
            ## Initialize object variables
            training_data = Data(nbags)
            training_data.set_domains_and_labels(originalBags)
            labeledPos = {}
            labeledNeg = {}
            for bag in range(nbags):
                labeledNeg[bag] = []
                labeledPos[bag] = []
            alignedBags = originalBags.copy()
            if smartguess:
                dataReplacer.setWeights(predictBegin(training_data, self.probs))
            weights = dataReplacer.getWeights()

            ## THE ULTIMATE LOOP
            for iteration in range(query_budget):
                    
                print("=================")
                print("ITERATION", iteration)
                print("=================")
                
                # Set the input for the normalizing flows.
                if not original:
                    dataReplacer.setData(originalBags)
                    if restart:
                        try:
                            os.remove(os.path.join(pathlib.Path().parent.resolve(), "alignflow_master", "ckpts", "normalaligner", "best.pth.tar"))
                        except Exception as exc:
                            print("No model found to remove")

                    # Align the instances
                    aligner = main(n_features, dataReplacer, y_inst, dataname + "."+ self.versionname + "." + str(experimentnumber), load)

                    # Get the aligned data
                    alignedData = dataReplacer.getLatent()
                    alignedBags = np.array(list(alignedData.values()))
                    weights = dataReplacer.getWeights()

                    # Change the data used in ALBA to the aligned data
                    for k, v in alignedData.items():
                        training_data.reset_domain(k, v)

                # Select an instance using ALBA
                #warnings.filterwarnings('ignore')
                queries = alba.fit_query(training_data, self.probs,  True)
                i = 0
                key,idx = queries[i]
                while dataBag.isLabeled(key,idx):
                    i+=1
                    key, idx = queries[i]
                instance = training_data.get_domain(key)[idx]

                ## Query the instance to the oracle (= get the label)
                lbl = dataBag.getLabel(key, idx)
                training_data.set_new_label(key, idx, lbl)
                training_data.set_last_labeled(key,idx)
                dataBag.label(key,idx)
                            
                # Store some information/bookkeeping to use for plotting later on
                rewardInfo.chooseArm(key)
                if (lbl == 1):
                    labeledPos[key].append(instance.tolist())
                else:
                    labeledNeg[key].append(instance.tolist())
                #performance.append(dataBag.measureAccuracy(predictions))

                # Plot the result of this iteration
                ws = currentPrediction(alba, training_data, alignedBags, key, instance, labeledPos,labeledNeg, iteration, weights = weights)

                if not original:
                    dataReplacer.setWeights(ws)
                    


        return rewardInfo, current


    def pplot(self, bags,probabilities, weights, instance, dataBag, t, scoreroc, scorepr, scorerocBag, scoreprBag):
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        fig, ax = plt.subplots( nrows=1, ncols=1 ,figsize=(16,9)) 
        domain = np.zeros((0,2))
        prs = np.zeros((0))
        w = np.zeros((0))
        fl1 = True
        fl2 = True
        for bag in range(len(bags)):
            domain = np.concatenate((domain,bags[bag]))
            prs = np.concatenate((prs, probabilities[bag]))
            w = np.concatenate((w, weights[bag]))

            anomalies = []
            normals = []
            wn = []
            wa = []
            prsa = []
            prsn = []

            domain = bags[bag]
            for idx in range(len(domain)):
                if dataBag.isAnomaly(bag, idx):
                    anomalies.append(domain[idx])
                    prsa.append(probabilities[bag][idx])
                    wa.append(weights[bag][idx])
                else:
                    normals.append(domain[idx])
                    prsn.append(probabilities[bag][idx])
                    wn.append(weights[bag][idx])

            normals = np.asarray(normals)
            anomalies = np.asarray(anomalies)
            wn = 1-np.asarray(wn)
            wa = 1-np.asarray(wa)
            if (len(normals)>0):
                if fl1:
                    z = ax.scatter(normals[:,0], normals[:,1], marker='o', norm=norm, c=prsn, cmap="coolwarm", s=250-200*wn, edgecolors='k', label = "Real normal")#, c= 'b')
                    fl1 = False
                else:
                    z = ax.scatter(normals[:,0], normals[:,1], marker='o', norm=norm, c=prsn, cmap="coolwarm", s=250-200*wn, edgecolors='k')#, c= 'b')
            if (len(anomalies)>0):
                if fl2:
                    ax.scatter(anomalies[:,0], anomalies[:,1],  marker='+', norm=norm, c=prsa, cmap="coolwarm", s=250-200*wa, edgecolors='k', label = "Real anomaly")#,c= 'b') 
                    fl2 = False
                else:
                    ax.scatter(anomalies[:,0], anomalies[:,1],  marker='+', norm=norm, c=prsa, cmap="coolwarm", s=250-200*wa, edgecolors='k')#,c= 'b')  
        
        ax.scatter(instance[0], instance[1],c='lime',label = "Selected instance",s=400, edgecolors='k')
        plt.title('Iteration '+str(t).zfill(3), fontsize = 16)
        
        textstr = "ROC AUC = {:.10f}\nPR AUC = {:.10f}\nROC AUC BAG= {:.10f}\nPR AUC BAG= {:.10f}".format(scoreroc, scorepr, scorerocBag, scoreprBag)
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=props)

        
        cbar = fig.colorbar(z,ax= ax, label ="higher score = more positive")
        tick_font_size = 14
        cbar.ax.tick_params(labelsize=tick_font_size)
        plt.rcParams.update({'font.size': 14})
        ax.legend(loc = "lower right", fontsize= 14)

        fig.savefig(os.path.join(current,'colorimg2/iteration'+str(t).zfill(3)+'.png'),bbox_inches='tight')
        plt.close(fig)