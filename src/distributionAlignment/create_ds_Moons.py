"""
This file implements an interface for creating a dataset consisting of 2 bags with each a different moon with some anomalies:

    gen_data(k, nbags, bag_contfactor = 0.1, seed = 331, maxAnoms = 1)
        -> bags, bags_labels, X_inst, y_inst
        Remark: nbags does not make a difference, the output is always 2 bags
"""

import pandas as pd
import numpy as np

from sklearn.datasets import make_moons


def gen_data(k, nbags, bag_contfactor = 0.1, seed = 331, maxAnoms = 1):
    np.random.seed(seed)
    xmin, xmax = -1.2,2.2
    ymin, ymax = -1.2,1.2
    
    bags_labels = np.ones(2, int)
    bags = {}
    X_inst = np.empty(shape = (0,2))
    y_inst = np.array([])
    flag = True
    for b in range(2):  
        if flag:
            cnt = maxAnoms
            flag = False
        else:
            if maxAnoms == 1:
                cnt = 1
            else:
                cnt = np.random.randint(1,maxAnoms)
        M,w = make_moons(2*(k-cnt), shuffle = True, noise = .15)
        
        if b == 0:
            #anomalies = np.concatenate((np.random.uniform(xmin, xmax, size = (cnt, 1)),np.random.uniform(ymin, ymax, size = (cnt, 1))), axis = 1)
            anomalies = np.concatenate((np.random.normal( loc = 0, scale = .3, size = (cnt, 1)),np.random.normal( loc = 0.25, scale = .3, size = (cnt, 1))), axis = 1)
        else:
            anomalies = np.concatenate((np.random.normal( loc = 1, scale = .3, size = (cnt, 1)),np.random.normal( loc = 0.25, scale = .3, size = (cnt, 1))), axis = 1)
        """for i in range(cnt):
            while ((anomalies[i,0]>mean[0]-2*std[0]) and (anomalies[i,0]<mean[0]+2*std[0])
                and (anomalies[i,1]>mean[1]-2*std[1]) and (anomalies[i,1]<mean[1]+2*std[1])):
                anomalies[i] = make_moons(2*(k-cnt), shuffle = True, noise = .2)"""
        X = M[w==b]
        y = np.zeros_like(w[w == b])
        X = np.concatenate((X, anomalies), axis = 0)
        y = 0*y
        y = np.concatenate((y, np.ones((cnt))))
        bags_labels[b] = 1

        bags[b] = X
        X_inst = np.concatenate((X_inst, X))
        y_inst = np.concatenate((y_inst, y))
    return bags, bags_labels, X_inst, y_inst

bags, bags_labels, X_inst, y_inst = gen_data(30,2)