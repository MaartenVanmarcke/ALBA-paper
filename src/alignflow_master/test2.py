import csv
import os
import pathlib
current = pathlib.Path().absolute()
with open(os.path.join(current, "src", "alignflow_master",'test2.csv')) as csvfile:
    file = csv.reader(csvfile, delimiter=',')
    i = -2
    n_dim = [[],[],[]]
    epoch0IF = [[],[],[]]
    epoch200Norm = [[],[],[]]
    for row in file:
        if i == -2:
            i += 1
            continue
        rr = row
        if int(rr[0]) == 2:
            i += 1 
        n_dim[i].append(int(rr[0]))
        epoch0IF[i].append(float(rr[2]))
        epoch200Norm[i].append(float(rr[3]))

print(n_dim)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(n_dim[0], epoch0IF[0], label = "Original: IF")
plt.plot(n_dim[0], epoch200Norm[0], label = "Aligned: -ll")
plt.title("Weights 1 / 0")
plt.xscale("log")
plt.xlabel("# Features")
plt.ylabel("AUCROC")
plt.legend()
plt.ylim([0,1.05])
plt.grid()
plt.show()

plt.figure()
plt.plot(n_dim[1], epoch0IF[1], label = "Original: IF")
plt.plot(n_dim[1], epoch200Norm[1], label = "Aligned: -ll")
plt.title("Weights .6 / .4")
plt.xscale("log")
plt.xlabel("# Features")
plt.ylabel("AUCROC")
plt.legend()
plt.ylim([0,1.05])
plt.grid()
plt.show()

plt.figure()
plt.plot(n_dim[2], epoch0IF[2], label = "Original: IF")
plt.plot(n_dim[2], epoch200Norm[2], label = "Aligned: -ll")
plt.title("Weights .6 / .4 & standardize")
plt.xscale("log")
plt.xlabel("# Features")
plt.ylabel("AUCROC")
plt.legend()
plt.ylim([0,1.05])
plt.grid()
plt.show()
