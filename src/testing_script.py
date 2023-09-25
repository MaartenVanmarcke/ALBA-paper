from data import Data
from methods import MABMethod
import numpy as np
import math
import matplotlib.pyplot as plt


## Define radius
def raddist(x,y):
    return math.sqrt((x-.5)**2+(y-.5)**2)
def isAnomaly(x,y):
    return (raddist(x,y)>.5)

## Create toy data set
training_data = Data(2)
domain_1 = np.random.default_rng(seed=125).random((35,2))
domain_2 = np.random.default_rng(seed=45).random((35,2))
anomalies_1 = []
anomalies_2 = []
normals_1 = []
normals_2 = []

for instance in domain_1:
    if isAnomaly(instance[0], instance[1]):
        anomalies_1.append(instance)
    else:
        normals_1.append(instance)

for instance in domain_2:
    if isAnomaly(instance[0], instance[1]):
        anomalies_2.append(instance)
    else:
        normals_2.append(instance)

anomalies_1 = np.asarray(anomalies_1)
anomalies_2 = np.asarray(anomalies_2)
normals_1 = np.asarray(normals_1)
normals_2 = np.asarray(normals_2)

plt.scatter(normals_1[:,0], normals_1[:,1], c= 'b')
plt.scatter(normals_2[:,0], normals_2[:,1], c= 'g')
if (len(anomalies_1)>0):
    plt.scatter(anomalies_1[:,0], anomalies_1[:,1], c= 'b', marker='+')
if (len(anomalies_2)>0):
    plt.scatter(anomalies_2[:,0], anomalies_2[:,1], c= 'g', marker='+')
plt.show()

## Create 2 domains
training_data.set_domains_and_labels({0:domain_1, 1:domain_2})
query_budget = 1
alba = MABMethod(mab="rotting-swa", query_budget=query_budget, verbose=True )

t = 0
while t<query_budget:   
    queries = alba.fit_query(training_data, True)
    instance = queries[0]
    print(instance)
    plt.scatter(normals_1[:,0], normals_1[:,1], c= 'b')
    plt.scatter(normals_2[:,0], normals_2[:,1], c= 'g')
    if (len(anomalies_1)>0):
        plt.scatter(anomalies_1[:,0], anomalies_1[:,1], c= 'b', marker='+')
    if (len(anomalies_2)>0):
        plt.scatter(anomalies_2[:,0], anomalies_2[:,1], c= 'g', marker='+')
    plt.scatter()
    plt.show()
    t += 1