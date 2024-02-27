"""
This file implements an interface for creating a dataset with bags consisting of multiple gaussians and some anomalies:

    gen_data(k, nbags, bag_contfactor = 0.1, seed = 331, maxAnoms = 1)
        -> bags, bags_labels, X_inst, y_inst
"""

import pandas as pd
import numpy as np

def gen_data(k, nbags, bag_contfactor = 0.1, seed = 331, maxAnoms = 1):
    np.random.seed(seed)
    norm_clusters = {1:([1,1],[2,1]), 2: ([-1,8], [1.25,3]), 3:([10,-1], [2,2]), 4:([5,-2], [1,2]),
                 5:([3,2], [1.5,2]), 6:([-8,0], [1,3]), 7:([-10,-5], [3,3]), 8:([0,0], [2,2]),
                 9:([5,-5], [3,1]), 10:([-5,5], [2,4])}
    anom_clusters = {11:([7,10],[2,2]), 12:([-3,-4],[1.5,1.5]), 13:([-10,12],[1.5,1.5]),
                     14:([-14,4],[.25,3])}
    
    n_norm_clusters = len(norm_clusters.keys())
    n_anom_clusters = len(anom_clusters.keys())
    tot_clusters = n_norm_clusters + n_anom_clusters
    
    bags_labels = np.zeros(nbags, int)
    bags = {}
    X_inst = np.empty(shape = (0,2))
    y_inst = np.array([])
    flag = True
    for b in range(nbags):
        label = np.random.binomial(1, bag_contfactor, size=1)[0]
        w = np.zeros(tot_clusters, float)
        if label == 0:
            w = np.zeros(n_norm_clusters, int)
            ## To get different distributions, uncomment:
            #w = np.zeros(n_norm_clusters, float)
            #tmp_norm_cls = int(np.round(np.random.uniform(low=0.5/n_norm_clusters,high=1.0,size=1)[0]*n_norm_clusters,0))
            #chosen_normcls = np.random.choice(np.arange(0,n_norm_clusters),tmp_norm_cls,replace=False)
            #w[chosen_normcls] = np.random.uniform(low=0.0, high=1.0, size=tmp_norm_cls)
            w[np.random.randint(0,len(w)-1)] = k
            X,y = gen_normals(norm_clusters, w)
        elif label == 1:
            if flag:
                cnt = maxAnoms
                flag = False
            else:
                if maxAnoms == 1:
                    cnt = 1
                else:
                    cnt = np.random.randint(1,maxAnoms)
                #cnt = np.random.randint(1,k/5)
            xmin, xmax = -15,15
            ymin, ymax = -9,9
            w = np.zeros(tot_clusters, int)
            normal = np.random.randint(0,n_norm_clusters-1)
            mean, std = norm_clusters[normal+1]
            w[normal] = k-cnt

            anomalies = np.concatenate((np.random.uniform(xmin, xmax, size = (cnt, 1)),np.random.uniform(ymin, ymax, size = (cnt, 1))), axis = 1)
            for i in range(cnt):
                while ((anomalies[i,0]>mean[0]-2*std[0]) and (anomalies[i,0]<mean[0]+2*std[0])
                       and (anomalies[i,1]>mean[1]-2*std[1]) and (anomalies[i,1]<mean[1]+2*std[1])):
                    anomalies[i] = np.concatenate((np.random.uniform(xmin, xmax, size = (1, 1)),np.random.uniform(ymin, ymax, size = (1, 1))), axis = 1)
            

            ''' while sum(w[-n_anom_clusters:]) != cnt:
                w = np.zeros(tot_clusters, int)
                chosen_anomcls = np.random.choice(np.arange(n_norm_clusters,tot_clusters),cnt,replace=True)
                w[np.random.randint(0,n_norm_clusters-1)] = k-cnt
                for j in chosen_anomcls:
                    w[j] = w[j]+1'''
            X,y = gen_anomalies(norm_clusters, anom_clusters, w)
            X = np.concatenate((X, anomalies.T), axis = 1)
            y = 0*y
            y = np.concatenate((y, np.ones((cnt))))
            bags_labels[b] = 1

        bags[b] = X.T
        X_inst = np.concatenate((X_inst, X.T))
        y_inst = np.concatenate((y_inst, y))
    return bags, bags_labels, X_inst, y_inst
def gen_normals(norm_clusters, w):

    X1 = np.array([])
    X2 = np.array([])

    for key,val in norm_clusters.items():
        X1_mean = val[0][0]
        X1_var = val[1][0]
        X2_mean = val[0][1]
        X2_var = val[1][1]
        
        X1 = np.concatenate((X1,np.random.normal(loc=X1_mean, scale=X1_var, size=w[key-1])))
        X2 = np.concatenate((X2,np.random.normal(loc=X2_mean, scale=X2_var, size=w[key-1])))

    X = np.array([X1,X2]).reshape(2,-1)
    y = np.zeros(sum(w), int)
    return X,y

def gen_anomalies(norm_clusters, anom_clusters, w):
    
    bag_clusters = {**norm_clusters, **anom_clusters}
    
    X1 = np.array([])
    X2 = np.array([])

    for key,val in bag_clusters.items():
        X1_mean = val[0][0]
        X1_var = val[1][0]
        X2_mean = val[0][1]
        X2_var = val[1][1]
        
        X1 = np.concatenate((X1,np.random.normal(loc=X1_mean, scale=X1_var, size=w[key-1])))
        X2 = np.concatenate((X2,np.random.normal(loc=X2_mean, scale=X2_var, size=w[key-1])))
    nnormals = sum(w[:-len(anom_clusters.keys())])
    nanom = sum(w[-len(anom_clusters.keys()):])
    y = np.zeros(nnormals+nanom, int)
    y[-nanom:] = 1
    X = np.array([X1,X2]).reshape(2,-1)
    return X,y
