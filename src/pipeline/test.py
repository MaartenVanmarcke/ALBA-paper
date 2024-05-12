import numpy as np
import matplotlib.pyplot as plt
from constructBags import ConstructBags
from preprocessor import Preprocessor

data = np.random.uniform(-10,10,size = (100,3))
anoms = np.random.uniform(-15,-10,size = (20,3))
bagCreator = ConstructBags(8,5,8,.75)
preprocessor = Preprocessor()
bags, bags_labels, y_inst = bagCreator.createBags(data, anoms)

print(bags_labels, y_inst)

fig = plt.figure()
ax = fig.add_subplot(1, 3, 1, projection='3d')
ax.scatter3D(data[:,0], data[:,1], data[:,2])
ax = fig.add_subplot(1, 3, 2, projection='3d')
for k,v in bags.items():
    ax.scatter3D(v[:,0], v[:,1], v[:,2], label = str(k))
plt.legend()
ax = fig.add_subplot(1, 3, 3, projection='3d')
bags, bags_labels, X_inst, y_inst = preprocessor.standardize(bags, bags_labels, y_inst)
for k,v in bags.items():
    ax.scatter3D(v[:,0], v[:,1], v[:,2], label = str(k))
plt.legend()
plt.show()