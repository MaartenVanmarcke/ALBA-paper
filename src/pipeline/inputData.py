import numpy as np
import os
import pathlib
current = pathlib.Path().parent.absolute()
p =  os.path.join(current, "src", "seed.txt")
file = open(p)
seed = int(file.read())
file.close()
np.random.seed(seed)

class InputData:
    def __init__(self, name) -> None:
        self.name = name
        self.loadData()

    def getName(self):
        return self.name

    def loadData(self):
        self.normals = None
        self.anomalies = None

    def getNormals(self):
        return self.normals
    
    def getAnomalies(self):
        return self.anomalies
    
    def getNumberOfFeatures(self):
        return 0
    
    def getNumberOfInstances(self):
        return 0
    
    def getNumberOfAnomalies(self):
        return 0
    
class TestData(InputData):
    def __init__(self) -> None:
        name = "TestData"
        super().__init__(name)

    def loadData(self):
        self.normals = np.random.uniform(-10,10,size = (100,2))
        self.anomalies = np.random.uniform(-15,-10,size = (20,2))
    
    def getNumberOfInstances(self):
        return 100
    
    def getNumberOfAnomalies(self):
        return 20
    
    def getNumberOfFeatures(self):
        return 2
    


class Speech_36(InputData):
    
    def __init__(self) -> None:
        name = "36_sspeech"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','36_speech.npz'), allow_pickle=True)
        data, labels = data['X'], data['y']
        idxs = labels == 0
        self.normals = data[idxs, :]
        self.anomalies = data[np.invert(idxs), :]
    
    def getNumberOfInstances(self):
        return len(self.normals)+len(self.anomalies) # 4819
    
    def getNumberOfAnomalies(self):
        return len(self.anomalies) # 257
    
    def getNumberOfFeatures(self):
        return len(self.normals[0,:]) # 5

class Annthyroid_2(InputData):
    
    def __init__(self) -> None:
        name = "2_annthyroid"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','13_fraud.npz'), allow_pickle=True)
        data, labels = data['X'], data['y']
        idxs = labels == 0
        self.normals = data[idxs, :]
        self.anomalies = data[np.invert(idxs), :]
    
    def getNumberOfInstances(self):
        return len(self.normals)+len(self.anomalies) # 4819
    
    def getNumberOfAnomalies(self):
        return len(self.anomalies) # 257
    
    def getNumberOfFeatures(self):
        return len(self.normals[0,:]) # 5
    
class nbr4(InputData):
    
    def __init__(self) -> None:
        name = "nbr4"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','19_landsat.npz'), allow_pickle=True)
        data, labels = data['X'], data['y']
        idxs = labels == 0
        self.normals = data[idxs, :]
        self.anomalies = data[np.invert(idxs), :]
    
    def getNumberOfInstances(self):
        return len(self.normals)+len(self.anomalies) # 4819
    
    def getNumberOfAnomalies(self):
        return len(self.anomalies) # 257
    
    def getNumberOfFeatures(self):
        return len(self.normals[0,:]) # 5
    

class Skin_33(InputData):
    
    def __init__(self) -> None:
        name = "33_skin"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','33_skin.npz'), allow_pickle=True)
        data, labels = data['X'], data['y']
        idxs = labels == 0
        self.normals = data[idxs, :]
        self.anomalies = data[np.invert(idxs), :]
    
    def getNumberOfInstances(self):
        return len(self.normals)+len(self.anomalies) # 4819
    
    def getNumberOfAnomalies(self):
        return len(self.anomalies) # 257
    
    def getNumberOfFeatures(self):
        return len(self.normals[0,:]) # 5
    

if __name__=="__main__":
    ff = Speech_36()