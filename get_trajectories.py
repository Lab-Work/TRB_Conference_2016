import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("classic")
#plt.style.use("seaborn-white")
import matplotlib
matplotlib.rc("font", family="FreeSans", size=16)
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
    phi = float(x)/dim[1]*2*np.pi + np.pi/2
    x_ = rho*np.cos(phi) + R
    y_ = rho*np.sin(phi) + R
    return np.array([x_, y_])

SRC = sys.argv[1]
NVEH = int(sys.argv[2])
print "Reading %s ..." % SRC
detections = np.load(SRC)
dim = detections[0]
r = dim[1]/(2*np.pi)
R = r + dim[0]

if os.path.isfile("results/dynamics_"+SRC[-6:-4]+".npy"):
    dynamics = np.load("results/dynamics_"+SRC[-6:-4]+".npy")
    print "Analyzing dynamics..."
else:
    dynamics = []
    for idx, frame in enumerate(detections):
        sys.stdout.write("Processing frame %d...\r" % idx)
        if idx > 30 and len(frame) > 100:
            for num, pixel in enumerate(frame):
                frame[num, 0:2] = Rec2Ann(pixel[1], pixel[0])

            mbk = MiniBatchKMeans(n_clusters=NVEH)
            mbk.fit(frame)
            labels = mbk.labels_.astype(np.int)
            labels_unique = np.unique(labels)
            
            speeds = []
            for lb in labels_unique:
                lb_mask = labels == lb
                lb_cluster = frame[lb_mask]
                count, bins = np.histogram(frame[:, 2], bins=25)
                lb_spd = (bins[np.argmax(count)]+
                          bins[np.argmax(count)+1])/2.
                speeds.append(lb_spd)

            centers = mbk.cluster_centers_[:,0:2]
            centers = np.array([[idx, c[0], c[1], s] 
                                for c, s in zip(centers, speeds)])
            for center in centers:
                dynamics.append(center)

    dynamics = np.array(dynamics)
    print "\nAnalyzing dynamics..."

if True:
    dbscan = DBSCAN(eps=36, min_samples=5)
    dbscan.fit(dynamics)
    labels = dbscan.labels_
    labels_unique = np.unique(labels)
else:
    connectivity = kneighbors_graph(dynamics, 
                                    n_neighbors=4, 
                                    include_self=False)
    agg = AgglomerativeClustering(n_clusters=NVEH, 
                                  connectivity=connectivity, 
                                  linkage="average")
    agg.fit(dynamics)
    labels = agg.labels_
    labels_unique = np.unique(labels)

trajectories = []
for lb in labels_unique:
    lb_mask = labels == lb
    lb_cluster = dynamics[lb_mask]
    print len(lb_cluster)
    if len(lb_cluster) > 30*100:
        trajectories.append(lb_cluster)
trajectories = np.array(trajectories)

print "Found %d trajectories." % trajectories.shape[0]
print "Saving raw dynamics to disk..."
np.save("results/dynamics_"+SRC[-6:-4], dynamics)
if trajectories.shape[0] != NVEH:
    np.save("results/dynamics_"+SRC[-6:-4], dynamics)
    print "Warning: there should be %d trajectoies." % NVEH
    print "Try to adjust parameters."
else:
    print "Saving trajectories to disk..."
    np.save("results/trajectories_"+SRC[-6:-4], trajectories)
print "Done."

colors = matplotlib.cm.jet(np.linspace(1,0,NVEH))
fig = plt.figure(figsize=(15,9))
ax1 = fig.add_subplot(1,2,1, projection="3d")
ax1.scatter(dynamics[:,1],dynamics[:,2],dynamics[:,0],s=1)
ax1.set_xlabel("X (pixels)")
ax1.set_ylabel("Y (pixels)")
ax1.set_zlabel("Time (1/30 sec)")
#ax1.set_title("Before Clustering")
ax1.axis("equal")
ax2 = fig.add_subplot(1,2,2, projection="3d")
for trajectory, c in zip(trajectories,colors):
    ax2.scatter(trajectory[:,1],trajectory[:,2],trajectory[:,0],s=1,color=c)
ax2.axis("equal")
ax2.set_xlabel("X (pixels)")
ax2.set_ylabel("Y (pixels)")
ax2.set_zlabel("Time (1/30 sec)")
#ax2.set_title("After Clustering")
fig.tight_layout()
plt.show()
#plt.savefig(SRC.replace(".npy",".pdf"))
