from typing import Any
from pipeline.mainMethod import MainMethod
from pipeline.mainMethodFlattened import MainMethodFlattened
import numpy as np

class Method:
    def __init__(self) -> None:
        self.name = "XXX"
        self.method = None

    def __call__(self, dataname, experimentnumber,  bags, bags_labels, X_inst, y_inst,*args: Any, **kwds: Any) -> Any:
        print("@@@@@@@@@@@é@@@@@@@@@@@é")
        print("METHOD:",self.method.versionname)
        print("@@@@@@@@@@@é@@@@@@@@@@@é")
        return self.method(dataname, experimentnumber, bags, bags_labels, X_inst, y_inst)

class WithoutAlignmentMethod(Method):
    def __init__(self) -> None:
        self.name = "WithoutAlignment"
        
    def __call__(self, dataname, query_budget, experimentnumber, bags, bags_labels, X_inst, y_inst, *args: Any, **kwds: Any) -> Any:
        self.method = MainMethod(versionname = self.name,
                                 original = True,
                                 al_strategy= "entropy",
                                 mabreward= "thesis",
                                 load = False,
                                 query_budget = query_budget,
                                 restart = False,
                                 smartguess = False,
                                 probs = "IF")
        return super().__call__(dataname, experimentnumber, bags, bags_labels, X_inst, y_inst,*args, **kwds)
    
class AlbaMethod(Method):
    def __init__(self) -> None:
        self.name = "AlbaMethod"
        
    def __call__(self, dataname, query_budget, experimentnumber, bags, bags_labels, X_inst, y_inst, *args: Any, **kwds: Any) -> Any:
        self.method = MainMethod(versionname = self.name,
                                 original = True,
                                 al_strategy= "random",
                                 mabreward= "cosine",
                                 load = False,
                                 query_budget = query_budget,
                                 restart = False,
                                 smartguess = False,
                                 probs = "IF")
        return super().__call__(dataname, experimentnumber, bags, bags_labels, X_inst, y_inst,*args, **kwds)
    
class SmartInitialGuessMethod(Method):
    def __init__(self) -> None:
        self.name = "SmartInitialGuess"
        
    def __call__(self, dataname, query_budget, experimentnumber, bags, bags_labels, X_inst, y_inst, *args: Any, **kwds: Any) -> Any:
        self.method = MainMethod(versionname = self.name,
                                 original = False,
                                 al_strategy= "entropy",
                                 mabreward= "thesis",
                                 load = False,
                                 query_budget = query_budget,
                                 restart = True,
                                 smartguess = True,
                                 probs = "loglike")
        return super().__call__(dataname, experimentnumber, bags, bags_labels, X_inst, y_inst,*args, **kwds)

class WithAlignmentMethod(Method):
    def __init__(self) -> None:
        self.name = "WithAlignment"
        
    def __call__(self, dataname, query_budget, experimentnumber, bags, bags_labels, X_inst, y_inst, *args: Any, **kwds: Any) -> Any:
        self.method = MainMethod(versionname = self.name,
                                 original = False,
                                 al_strategy= "entropy",
                                 mabreward= "thesis",
                                 load = False,
                                 query_budget = query_budget,
                                 restart = True,
                                 smartguess = False,
                                 probs = "loglike")
        return super().__call__(dataname, experimentnumber, bags, bags_labels, X_inst, y_inst,*args, **kwds)
    
class ActiveLearning(Method):
    def __init__(self) -> None:
        self.name = "BasicActiveLearning"
        
    def __call__(self, dataname, query_budget, experimentnumber, bags, bags_labels, X_inst, y_inst, *args: Any, **kwds: Any) -> Any:
        self.method = MainMethodFlattened(versionname = self.name,
                                 original = True,
                                 al_strategy= "entropy",
                                 mabreward= "thesis",
                                 load = False,
                                 query_budget = query_budget,
                                 restart = True,
                                 smartguess = False,
                                 probs = "IF")
        return super().__call__(dataname, experimentnumber, bags, bags_labels, X_inst, y_inst,*args, **kwds)
    
class RandomSampling(Method):
    def __init__(self) -> None:
        self.name = "RandomSampling"
        
    def __call__(self, dataname, query_budget, experimentnumber, bags, bags_labels, X_inst, y_inst, *args: Any, **kwds: Any) -> Any:
        self.method = MainMethodFlattened(versionname = self.name,
                                 original = True,
                                 al_strategy= "random",
                                 mabreward= "thesis",
                                 load = False,
                                 query_budget = query_budget,
                                 restart = True,
                                 smartguess = False,
                                 probs = "IF")
        return super().__call__(dataname, experimentnumber, bags, bags_labels, X_inst, y_inst,*args, **kwds)

