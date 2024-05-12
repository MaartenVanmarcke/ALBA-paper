import pandas as pd
import numpy as np

def gen_data(k, nbags, bag_contfactor = 0.1, seed = 331):
    np.random.seed(seed)
    n_points = 5

    bags_labels = np.random.binomial(1, bag_contfactor, size=nbags)
    bags = {}
    X_inst = np.empty(shape = (0,2))
    y_inst = np.array([])
    
    points = np.zeros((n_points,2))
    radii = np.random.uniform(2,4,size = ((n_points)))
    angles = np.random.uniform(0,2*np.pi,size = ((n_points)))
  
    for i in range(1, n_points):
        aa = angles[i-1]
        points[i,:] = points[i-1,:] + radii[i-1]*np.array([np.cos(aa), np.sin(aa)]).T

    for b in range(nbags):
        label = bool(bags_labels[b])
        chosen_points = np.random.randint(0,n_points, (k))
        data = np.zeros((k,2))
        labels = np.zeros((k))

        data = np.random.normal(loc = points[chosen_points,:], scale = 1 , size = (k,2))

        if label:
            n_anomalies = np.random.randint(1,.15*k)
            labels[-n_anomalies:] = 1

            chosen_points = np.random.randint(0,n_points, n_anomalies)
            radii = np.random.uniform(2,3.5,size = ((n_anomalies)))
            angles = np.random.uniform(0,2*np.pi,size = ((n_anomalies)))

            for i in range(1,n_anomalies+1):
                aa = angles[i-1]
                data[-i, :] = points[chosen_points[i-1],:] + radii[i-1]*np.array([np.cos(aa), np.sin(aa)]).T
                while(np.any(np.linalg.norm(points-data[-i, :],axis=1)< 1.5)):
                    choi = np.random.randint(0,n_points)
                    ra = np.random.uniform(2,3.5)
                    aa = np.random.uniform(0,2*np.pi)
                    data[-i, :] = points[choi,:]+ra*np.array([np.cos(aa), np.sin(aa)]).T

        X_inst = np.concatenate((X_inst, data))
        y_inst = np.concatenate((y_inst, labels))
        bags[b] = data

    return bags, bags_labels, X_inst, y_inst

if __name__=="__main__":
    import matplotlib.pyplot as plt
    colors = ['b','g','r','c','m','k','y', 'lime','deeppink','aqua','yellow','gray','darkorange','saddlebrown','salmon']
    fig, axes = plt.subplots(3,4,figsize = (16,9))
    axes = np.array([axes[0,0],axes[0,1],axes[0,2],axes[0,3],axes[1,0],axes[1,1],axes[1,2], axes[1,3],axes[2,0],axes[2,1],axes[2,2]])
    ax = axes[-1]
    k = 30
    bags, bags_labels, X_inst, y_inst = gen_data(k, 10, 0.4, 1302)
    markers = np.zeros_like(y_inst, dtype = str)
    for b in bags.keys():
        idxs = y_inst[b*k:(b+1)*k] == 1
        ax.scatter(bags[b][idxs,0], bags[b][idxs,1], s = 250, c = colors[b], marker = "+", label = f"Anomalies {b}")
        axes[b].scatter(bags[b][idxs,0], bags[b][idxs,1], s = 250, marker = "+", c = colors[b], label = f"Anomalies {b}")
        idxs = np.invert(idxs)
        ax.scatter(bags[b][idxs,0], bags[b][idxs,1], s = 250, marker = ".", c = colors[b], label = f"Bag {b}")
        axes[b].scatter(bags[b][idxs,0], bags[b][idxs,1], s = 250, marker = ".", c = colors[b], label = f"Bag {b}")
    #plt.legend()
    for ax in axes:
        ax.set_aspect('equal', 'box')
    plt.show()

