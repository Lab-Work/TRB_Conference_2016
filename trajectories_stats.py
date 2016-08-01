import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("seaborn-deep")
#plt.style.use("seaborn-white")
import matplotlib
matplotlib.rc("font", family="FreeSans", size=14)
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.cluster import DBSCAN, MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.cluster import MiniBatchKMeans, KMeans, AffinityPropagation
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import sys
import os.path
import warnings
warnings.filterwarnings("ignore")

def Rec2Ann(x, y):
    rho = y + r
    phi = float(x)/dim[1]*2*np.pi
    x_ = rho*np.cos(phi) + R
    y_ = rho*np.sin(phi) + R
    return np.array([x_, y_])

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([rho, phi])

def find_centers(X, labels):
    labels_unique = np.unique(labels)
    centers = []
    for lb in labels_unique:
        lb_mask = labels == lb
        lb_cluster = X[lb_mask]
        x = np.sum(lb_cluster[:, 0]).astype(float)/len(lb_cluster)
        y = np.sum(lb_cluster[:, 1]).astype(float)/len(lb_cluster)
        centers.append([x, y])
    return np.array(centers)

SRC = sys.argv[1]
NVEH = int(sys.argv[2])
print "Reading %s ..." % SRC
detections = np.load(SRC)
duration = len(detections)
dim = detections[0]
r = dim[1]/(2*np.pi)
R = r + dim[0]

dynamics = np.load("results/dynamics_"+SRC[-6:-4]+".npy")
print "Analyzing dynamics..."

for idx, d in enumerate(dynamics):
    dynamics[idx, 1:3] = cart2pol(d[1]-R, d[2]-R)


# Calcualting headway
headway = []
for t, point in enumerate(dynamics):
    if abs(point[2] - int(point[2])) < 0.0025 and int(point[2]) != 0:
        headway.append(point)
headway = np.array(headway)
tmp1 = []
for rad in [-3,-2,-1,1,2,3]:
    tmp2 = []
    for t, h in enumerate(headway):
        if int(h[2]) == rad:
            tmp2.append(h[0])
    tmp2 = np.array(tmp2)
    tmp2.sort()
    tmp1.append((tmp2[1:]-tmp2[:-1])/30)
tmp1 = np.array(tmp1)
headway = np.array([h for rad in tmp1 for h in rad])
headway = headway[headway > 1]
headway = headway[headway < 10]
print "Avg headway is %.2f sec." % np.mean(headway)
print "Headway stdev is %.2f sec." % np.std(headway)
plt.figure()
plt.hist(headway, bins=20)
plt.xlabel("Headway (sec)")
plt.ylabel("Count")
plt.title("Headway Distribution")


# Calculating spacing
spacing = []
for t, point in enumerate(dynamics):
    if point[0] in np.arange(0,duration,300):
        spacing.append(point)
tmp1 = []
for time in np.arange(0,duration,300):
    tmp2 = []
    for t, s in enumerate(spacing):
        if s[0] == time:
            #tmp2.append(s[2])
            tmp2.append(s[2]/(2*np.pi)*3840*10.45*NVEH/3840)
    tmp2 = np.array(tmp2)
    tmp2.sort()
    tmp1.append(tmp2[1:]-tmp2[:-1])
tmp1 = np.array(tmp1)
spacing = np.array([s for time in tmp1 for s in time])
print "Avg spacing is %.2f m." % np.mean(spacing)
print "Spacing stdev is %.2f m." % np.std(spacing)
plt.figure()
plt.hist(spacing, bins=20)
plt.xlabel("Spacing (m)")
plt.ylabel("Count")
plt.title("Spacing Distribution")
plt.show()
