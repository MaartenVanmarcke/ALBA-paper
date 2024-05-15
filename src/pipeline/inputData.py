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
        print("DATASET:", name)
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
    


class ALOI_1(InputData):
    
    def __init__(self) -> None:
        name = "1_ALOI"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','1_ALOI.npz'), allow_pickle=True)
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
        data = np.load(os.path.join(current, "src","pipeline",'adbench','2_annthyroid.npz'), allow_pickle=True)
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
    
class Cardiotocography_7(InputData):
    
    def __init__(self) -> None:
        name = "7_Cardiotocography"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','7_Cardiotocography.npz'), allow_pickle=True)
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
    

class Celeba_8(InputData):
    
    def __init__(self) -> None:
        name = "8_celeba"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','8_celeba.npz'), allow_pickle=True)
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
    
class Letter_20(InputData):
    
    def __init__(self) -> None:
        name = "20_letter"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','20_letter.npz'), allow_pickle=True)
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
    
class SpamBase_35(InputData):
    
    def __init__(self) -> None:
        name = "35_SpamBase"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','35_SpamBase.npz'), allow_pickle=True)
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
    
class Speech_36(InputData):
    
    def __init__(self) -> None:
        name = "36_speech"
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
    
class Yeast_47(InputData):
    
    def __init__(self) -> None:
        name = "47_yeast"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','47_yeast.npz'), allow_pickle=True)
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
    ff = Celeba_8()