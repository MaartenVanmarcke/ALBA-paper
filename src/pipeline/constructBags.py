## inspiration from: https://realpython.com/k-means-clustering-python/

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings
from k_means_constrained import KMeansConstrained
class ConstructBags:

    def __init__(self, nclusters, nbags, instances_per_bag, bag_contfactor = .3) -> None:
        raise Exception("CONSTRUCTBAGS CANNOT BE USED!!")
        self.nclusters = nclusters
        self.nbags = nbags
        self.instances_per_bag = instances_per_bag
        self.bag_contfactor = bag_contfactor

    def _clusterNormals(self, normals,kk):
        ## Scaling before clustering needed?

        kmeans = KMeans(
            init="random", #"k-means++"
            n_clusters=self.nclusters,
            #n_init = 10,
            #max_iter=300,
            random_state=kk
        )

        kmeans.fit(normals)
        assignment = kmeans.labels_

        clusters = {}
        for i in range(self.nclusters):
            clusters[i] = []
        
        for i in range(len(assignment)):
            clusters[assignment[i]].append(normals[i,:])

        for i in range(self.nclusters):
            clusters[i] = np.asarray(clusters[i])
            if len(clusters[i]) < self.instances_per_bag:
                clusters.pop(i)

        newClusters = {}
        j = 0
        for key in clusters.keys():
            newClusters[j] = clusters[key]
            j += 1
        
        self.nclusters = len(clusters.keys())
        print("CLUSTERS:",self.nclusters)

        return newClusters
    
    def createBags(self, normals, anomalies,seed):
        np.random.seed(seed)
        scaler = StandardScaler()
        scaler = scaler.fit(np.concatenate((normals, anomalies)))
        normals = scaler.transform(normals)
        anomalies = scaler.transform(anomalies)

        clusters = self._clusterNormals(normals,seed)
        cluster_idxs = np.random.randint(0,self.nclusters, size = (self.nbags))
        
        bags = {}
        bags_labels = np.zeros((self.nbags))
        y_inst = np.zeros((0))
        for i in range(self.nbags):
            label = np.random.binomial(1, self.bag_contfactor, size=1)[0]
            if i == self.nbags-1 and np.all(bags_labels == 0):
                label = 1

            n_normals = self.instances_per_bag
            n_anomalies = 0
            if label == 1:
                ## [k/2] is kinda high...
                n_anomalies = np.random.randint(1, int(np.ceil(self.instances_per_bag/2)))
                n_normals = self.instances_per_bag - n_anomalies

            try:
                indices = np.random.choice(clusters[cluster_idxs[i]].shape[0], n_normals, replace=False)
            except ValueError as exc:
                if clusters[cluster_idxs[i]].shape[0] < n_normals:
                    print("Too little instances in this cluster to fill the bag.")
                    n_normals = clusters[cluster_idxs[i]].shape[0]
                    indices = np.asarray(range(n_normals))
                else:
                    raise exc
            bags[i] = clusters[cluster_idxs[i]][indices]

            
            anomaly_indices = np.random.choice(anomalies.shape[0], n_anomalies, replace=False)
            bags[i] = np.vstack((bags[i], anomalies[anomaly_indices]))
            bags_labels[i] = label
            y_inst = np.hstack((y_inst, np.zeros((n_normals)), np.ones((n_anomalies))))

        return bags, bags_labels, y_inst


class ConstructBagsWithoutClustering:

    def __init__(self, nclusters, nbags, instances_per_bag, bag_contfactor = .3) -> None:
        self.nclusters = nclusters
        self.nbags = nbags
        self.instances_per_bag = instances_per_bag
        self.bag_contfactor = bag_contfactor
    
    def createBags(self, normals, anomalies,seed):
        np.random.seed(seed)
        scaler = StandardScaler()
        scaler = scaler.fit(np.concatenate((normals, anomalies)))
        normals = scaler.transform(normals)
        anomalies = scaler.transform(anomalies)
        instanceIdxs = np.random.choice(normals.shape[0], size = (self.nbags, self.instances_per_bag), replace = False)
        
        bags = {}
        bags_labels = np.zeros((self.nbags))
        y_inst = np.zeros((0))
        for i in range(self.nbags):
            label = np.random.binomial(1, self.bag_contfactor, size=1)[0]
            if i == self.nbags-1 and np.all(bags_labels == 0):
                label = 1
            n_normals = self.instances_per_bag
            n_anomalies = 0
            bags[i] = normals[instanceIdxs[i,:],:]
            if label == 1:
                ## [k/2] is kinda high...
                n_anomalies = np.random.randint(1, int(np.ceil(self.instances_per_bag/2)))
                anomaly_indices = np.random.choice(anomalies.shape[0], size = (n_anomalies), replace=False)
                bags[i][-n_anomalies:,:] = anomalies[anomaly_indices,:]
                n_normals -= n_anomalies
            bags_labels[i] = label
            y_inst = np.hstack((y_inst, np.zeros((n_normals)), np.ones((n_anomalies))))

        return bags, bags_labels, y_inst


class ConstructBagsOpt:

    def __init__(self, nclusters, nbags, instances_per_bag, bag_contfactor = .3) -> None:
        self.nclusters = nclusters
        self.nbags = nbags
        self.instances_per_bag = instances_per_bag
        self.bag_contfactor = bag_contfactor

    def _clusterNormals(self, normals,kk):
        ## Scaling before clustering needed?
        scaler = StandardScaler()
        normalstransf = scaler.fit_transform(normals)

        kmeans = KMeansConstrained(
                n_clusters=self.nclusters,
                size_min=self.instances_per_bag,
                random_state=kk
            )

        kmeans.fit(normalstransf)
        assignment = kmeans.labels_

        clusters = {}
        for i in range(self.nclusters):
            clusters[i] = []
        
        for i in range(len(assignment)):
            clusters[assignment[i]].append(normals[i,:])

        for i in range(self.nclusters):
            clusters[i] = np.asarray(clusters[i])
            if len(clusters[i]) < self.instances_per_bag:
                clusters.pop(i)

        newClusters = {}
        j = 0
        for key in clusters.keys():
            newClusters[j] = clusters[key]
            j += 1
        
        self.nclusters = len(clusters.keys())
        print("CLUSTERS:",self.nclusters)

        return newClusters
    
    def createBags(self, normals, anomalies,seed):
        np.random.seed(seed)
        clusters = self._clusterNormals(normals,seed)
        cluster_idxs = np.arange(0, self.nclusters)#np.random.randint(0,self.nclusters, size = (self.nbags))
        
        bags = {}
        bags_labels = np.zeros((self.nbags))
        y_inst = np.zeros((0))
        for i in range(self.nbags):
            label = np.random.binomial(1, self.bag_contfactor, size=1)[0]
            if i == self.nbags-1 and np.all(bags_labels == 0):
                label = 1

            n_normals = self.instances_per_bag
            n_anomalies = 0
            if label == 1:
                ## [k/2] is kinda high...
                n_anomalies = np.random.randint(1, int(np.ceil(self.instances_per_bag/2)))
                n_normals = self.instances_per_bag - n_anomalies

            try:
                indices = np.random.choice(clusters[cluster_idxs[i]].shape[0], n_normals, replace=False)
            except ValueError as exc:
                if clusters[cluster_idxs[i]].shape[0] < n_normals:
                    print("Too little instances in this cluster to fill the bag.")
                    n_normals = clusters[cluster_idxs[i]].shape[0]
                    indices = np.asarray(range(n_normals))
                else:
                    raise exc
            bags[i] = clusters[cluster_idxs[i]][indices]

            
            anomaly_indices = np.random.choice(anomalies.shape[0], n_anomalies, replace=False)
            bags[i] = np.vstack((bags[i], anomalies[anomaly_indices]))
            bags_labels[i] = label
            y_inst = np.hstack((y_inst, np.zeros((n_normals)), np.ones((n_anomalies))))

        return bags, bags_labels, y_inst

