from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection

class Preprocessor:

    def __init__(self) -> None:
        pass

    def standardize(self, bags, bags_labels, y_inst):
        X_inst = np.zeros((0,bags[0].shape[1]))
        for k, v in bags.items():
            X_inst = np.vstack((X_inst, v))
            
        scaler = StandardScaler().fit(X_inst)
        X_inst = scaler.transform(X_inst)
        newbags = {}
        ll = 0
        for k, v in bags.items():
            newbags[k] = scaler.transform(v)
            if not np.all(newbags[k] == X_inst[ll:ll+len(newbags[k])]):
                raise Exception("Implementation Error!!!")
            ll+=len(newbags[k])



        return newbags, bags_labels, X_inst, y_inst
    

class PreprocessorWithProjection:

    def __init__(self) -> None:
        self.dim = "auto"
        pass

    def standardize(self, bags, bags_labels, y_inst):
        newbags = {}
        X_inst = np.zeros((0,bags[0].shape[1]))
        for k, v in bags.items():
            transformer = SparseRandomProjection(n_components=self.dim)
            newbags[k] = transformer.fit_transform(v)
            self.dim = newbags[k].shape[1]
            
            
            scaler = StandardScaler()
            newbags[k] = scaler.fit_transform(newbags[k] )
            X_inst = np.vstack((X_inst, newbags[k]))



        self.dim = "auto"
        return newbags, bags_labels, X_inst, y_inst