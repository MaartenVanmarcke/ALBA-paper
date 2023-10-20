import numpy as np
import matplotlib.pyplot as plt
probabilities = np.arange(0,1,.01)
fig, ax = plt.subplots( nrows=1, ncols=1 ,figsize=(16,9)) 
   
ax.scatter(probabilities,probabilities, c=probabilities, cmap='gray')
plt.show()