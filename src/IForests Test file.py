import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltclr
from pyod.models.iforest import IForest


## PARAMS
cont_factor = .1
renormalize = True

## TRAIN DATA 
np.random.seed(1302)
n = 100
data = np.random.uniform(low = -8, high = 8, size = (n,2))

## TEST DATA 
x = np.arange(start = -8, stop = 8,step=.03)
X = np.meshgrid(x,x)
X,Y = np.ndarray.flatten(X[0]), np.ndarray.flatten(X[1])
test_data = np.zeros((len(X),2))
test_data[:,0] = X
test_data[:,1] = Y

## CLASSIFIER
clf = IForest(contamination = cont_factor)
clf.fit(data)
ss = clf.decision_function(data)
minim, maxim = np.min(ss), np.max(ss)
threshold = (clf.threshold_-minim)/(maxim-minim)
ss = (ss-minim)/(maxim-minim)
probabs = clf.predict_proba(data, method = "linear")[:,1]
#print( abs(ss-sss) < 0.000000000000001, [(ss[i], sss[i]) for i in range(len(ss)) if abs(ss[i]-sss[i])<.000000000000001])
preds = clf.predict(data)

## PLOT
clrs = pltclr.Normalize(vmin=0, vmax=1)
testpreds = clf.predict_proba(test_data, method = "linear")[:,1]
thresholds =  np.abs(testpreds - threshold) < .005

## RENORMALIZE
def _renorm(scores, threshold):
    data = scores/threshold
    return 1-np.power(2, -data)
testpreds = _renorm(testpreds, threshold)
probabs = _renorm(probabs, threshold)

fig, ax = plt.subplots( nrows=1, ncols=1, figsize = (16,16)) 
z = ax.imshow(np.reshape(testpreds, (int(np.sqrt(len(testpreds))),int(np.sqrt(len(testpreds))))),norm = clrs,cmap="coolwarm",
              origin = "lower", extent = [-8,8,-8,8])
plt.scatter(data[preds<1,0], data[preds<1,1], c = 'white', s = 200, label = "normals")
plt.scatter(data[preds>0,0], data[preds>0,1], c = "black",  marker = "+", s = 200, label = "anomalies")

cbar = fig.colorbar(z,ax= ax, label ="higher score = more positive")
tick_font_size = 14
cbar.ax.tick_params(labelsize=tick_font_size)
plt.rcParams.update({'font.size': 14})
textstr = "Contamination factor = {:.3f}\nThreshold = {:.3f}".format(cont_factor, threshold) 
# these are matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# place a text box in upper left in axes coords
ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', horizontalalignment='right', bbox=props)
ax.legend(loc = "upper left", fontsize= 14)
ax.set_xlim([-8,8])
ax.set_ylim([-8,8])
ax.set_title("Anomaly probabilities")
plt.show()