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
    


class Waveform_41(InputData):
    
    def __init__(self) -> None:
        name = "41_Waveform"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','41_Waveform.npz'), allow_pickle=True)
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
    



class Fault_12(InputData):
    
    def __init__(self) -> None:
        name = "12_Fault"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','12_fault.npz'), allow_pickle=True)
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

class Vowels_40(InputData):
    
    def __init__(self) -> None:
        name = "40_Vowels"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','40_vowels.npz'), allow_pickle=True)
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
    
class Cardio_6(InputData):
    
    def __init__(self) -> None:
        raise Exception("Unused dataset!")
        name = "6_cardio"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','6_cardio.npz'), allow_pickle=True)
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
   
    
    
class checkData(InputData):
    
    def __init__(self) -> None:
        name = "40_vowels"
        self.name = name 
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench',self.name+'.npz'), allow_pickle=True)
        data, labels = data['X'], data['y']
        idxs = labels == 0
        self.normals = data[idxs, :]
        self.anomalies = data[np.invert(idxs), :]
        print(self.getNumberOfFeatures())
    
    def getNumberOfInstances(self):
        return len(self.normals)+len(self.anomalies) # 4819
    
    def getNumberOfAnomalies(self):
        return len(self.anomalies) # 257
    
    def getNumberOfFeatures(self):
        return len(self.normals[0,:]) # 5








#######################


class Letter_20_Equal(InputData):
    
    def __init__(self) -> None:
        name = "20_Letter_equal_distr"
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
    
class Yeast_47_Equal(InputData):
    
    def __init__(self) -> None:
        name = "47_yeast_equal_distr"
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
    

class Vowels_40_Equal(InputData):
    
    def __init__(self) -> None:
        name = "40_vowels_equal_distr"
        super().__init__(name)

    def loadData(self):
        data = np.load(os.path.join(current, "src","pipeline",'adbench','40_vowels.npz'), allow_pickle=True)
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
    