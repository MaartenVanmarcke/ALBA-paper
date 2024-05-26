import numpy as np
import matplotlib.pyplot as plt
from constructBags import ConstructBags
from preprocessor import Preprocessor

from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans

np.random.seed(1302)
data = np.zeros((100,2))
xx = 70
data[:xx,:] = np.random.normal(-3,1,size = (xx,2))
data[xx:,:] = np.random.normal(3,1,size = (100-xx,2))

kmeansconstr = KMeansConstrained(
                n_clusters=2,
                size_min=50,
                random_state=1302
)
kmeans = KMeans(
            init="random", #"k-means++"
            n_clusters=2,
            #n_init = 10,
            #max_iter=300,
            random_state=1302)

labels_constr = kmeansconstr.fit_predict(data)
labels_nonconstr = kmeans.fit_predict(data)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
for i in range(len(labels_constr)):
    if labels_constr[i]  == 0:
        ax.scatter(data[i,0], data[i,1], c = "red")
    else:
        ax.scatter(data[i,0], data[i,1], c = "blue")
ax.set_title(str(np.count_nonzero(labels_constr))+"/"+str(100-np.count_nonzero(labels_constr)))
ax = fig.add_subplot(1, 2, 2)
for i in range(len(labels_nonconstr)):
    if labels_nonconstr[i]  == 0:
        ax.scatter(data[i,0], data[i,1], c = "red")
    else:
        ax.scatter(data[i,0], data[i,1], c = "blue")

ax.set_title(str(np.count_nonzero(labels_nonconstr))+"/"+str(100-np.count_nonzero(labels_nonconstr)))
plt.show()