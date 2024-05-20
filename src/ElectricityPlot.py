import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1303)
stepsize = .1
x = np.arange(0,24,stepsize)
y = np.zeros_like(x)
y[x<7] = 500
y[x>=7] = (1000-500)/(10-7)*(x[x>=7]-7)+500
y[x>=10] = y[x==10]
y[x>=12] = (850-1000)/(13-12)*(x[x>=12]-12)+y[x==10]
y[x>=13] = y[x==13]
y[x>=16] = (600-850)/(17.75-16)*(x[x>=16]-16)+y[x==16]
y[x>17.75] = 600
dev = np.random.normal(loc = 0, scale = 30, size = y.shape)
dev[x>7.5] = np.random.normal(loc = 0, scale = 60, size = dev[x>7.5].shape)
dev[x>16.5] = np.random.normal(loc = 0, scale = 30, size = dev[x>16.5].shape)
y = y + dev
dummy = y[np.logical_and(x>22.2, x<22.55)]
dummy = 900 +np.random.normal(loc = 0, scale = 60, size = dummy.shape)
y[np.logical_and(x>22.2, x<22.55)] = dummy
xxx = x[np.logical_and(x>22.2, x<22.55)]

plt.figure()
plt.plot(x, y)
plt.plot(xxx,dummy, c = "red")#, marker = "+")
plt.ylim([0,1200])
plt.xlim([0,24])
plt.xlabel("Time (h)", fontsize=13)
plt.ylabel("Power consumption (W)", fontsize=13)
plt.title("Electricity consumption pattern", fontsize=15)
plt.grid()
plt.savefig("Electricity1.png",bbox_inches='tight')
plt.show()
plt.close()

plt.figure()
xx = np.arange(0,1,stepsize)
for i in range(24):
    dummy = None
    dummy = y[np.logical_and(x>= i, x<i+1)]
    if i == 22:
        mm = dummy
        plt.plot(xx,dummy, c = "blue", label = "Hour "+str(i+1))
    else:
        plt.plot(xx,dummy, c = "grey", label = "Hour "+str(i+1))
plt.plot(xx,mm, c = "blue", linewidth=3, label = "Hour "+str(i+1))
plt.ylim([0,1200])
plt.xlim([0,.9])
plt.xlabel("Time (h)", fontsize=13)
plt.ylabel("Power consumption (W)", fontsize=13)
plt.title("Electricity consumption during anomalous hour", fontsize=15)
plt.grid()
plt.savefig("Electricity2.png",bbox_inches='tight')
plt.show()
plt.close()

plt.figure()
xx = np.arange(0,1,stepsize)
for i in range(24):
    dummy = None
    dummy = y[np.logical_and(x>= i, x<i+1)]
    dummy = (dummy-np.mean(dummy))/np.std(dummy)
    if i == 22:
        plt.plot(xx,dummy, c = "blue", label = "Hour "+str(i+1))
    else:
        plt.plot(xx,dummy, c = "k", label = "Hour "+str(i+1))
plt.ylim([-3,3])
plt.show()
plt.close()