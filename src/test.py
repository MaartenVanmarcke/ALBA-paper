import numpy as np

x = [2,3,4,5,6]
withoutAlignmentNorm = [.6383,.6367,.5883,.6483,.65167,0.5683333333]
withoutAlignmentIF = [.689167,.623,.7433,.5367,.653,0.6000000000]
withAlignmentNorm = [.89,1,1,1,1,1]
withAlignmentIF = [.8867,.9967,1,1,1,1]

import matplotlib.pyplot as plt
plt.figure()
plt.plot(x, withoutAlignmentNorm, label = "withoutAlignmentNorm")
plt.plot(x, withoutAlignmentIF, label = "withoutAlignmentIF")
plt.plot(x, withAlignmentNorm, label = "withAlignmentNorm")
plt.plot(x, withAlignmentIF, label = "withAlignmentIF")
plt.show()
plt.close()