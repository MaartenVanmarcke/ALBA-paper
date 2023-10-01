import numpy as np
from math import floor

def gen_simple_data(k, nbags, bag_contfactor = 0.1, seed = 331):
    np.random.seed(seed)
    nbrgauss = floor(nbags*k*(1-bag_contfactor))
    X1 = np.random.normal(2.0, 1, size=((nbrgauss,2)))
    X1 = np.append(X1,np.zeros((nbrgauss,1)),1)
    nbruniform = nbags*k-nbrgauss
    X2 = np.random.uniform(0.0,4.0, size = ((nbruniform,2)))
    X2 = np.append(X2,np.ones((nbruniform,1)),1)
    X = np.append(X2[:nbruniform//2,:],X1,0)
    X = np.append(X, X2[nbruniform//2:,:], 0)
    X_inst = X[:,:-1]
    y_inst = X[:,2]
    bags = np.zeros((nbags, k, 2))
    for bag in range(nbags):
        bags[bag] = X_inst[k*bag:(bag+1)*k,:]
    bags_labels = np.zeros((nbags,k))
    for bag in range(nbags):
        bags_labels[bag] = y_inst[k*bag:(bag+1)*k]

    return bags, bags_labels, X_inst, y_inst


gen_simple_data(10,2,0.1)
