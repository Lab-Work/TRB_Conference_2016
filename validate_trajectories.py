import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("ggplot")
import matplotlib
matplotlib.rc("font", family="FreeSans", size=16)
import scipy

SRC = "results/trajectories_04.npy"
detections = np.load(SRC.replace("trajectories", "detections"))
dim = detections[0]
r = dim[1]/(2*np.pi)
R = r + dim[0]

labels_ = np.load(SRC.replace(".npy", "_labels.npy"))
labels_ = labels_[5:-5]
labels = [[t*30, labels_[t,v,1], labels_[t,v,2]] 
          for t in range(labels_.shape[0])
          for v in range(labels_.shape[1])]
labels = np.asarray(labels)

estimates_ = np.load(SRC)
estimates = [[e[0], e[1]-R, e[2]-R]
             for estimate in estimates_ 
             for e in estimate
             if e[0] % 30 in [0, 15]]
estimates = np.asarray(estimates)

if False:
    distances = []
    for time in range(labels_.shape[0]):
        lb_data = []
        for lb in labels:
            if lb[0] == time*30:
                lb_data.append(lb[1:])
        est_data = []
        for est in estimates:
            if est[0] - time*30 in np.arange(-30,30):
                est_data.append(est[1:])
        if len(est_data) > 20:
            nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(est_data)
            distance, indices = nbrs.kneighbors(lb_data)
            distances.append(distance*0.0272)
    distances = np.asarray(distances).flatten()
    distances = distances[distances < 2]
    print distances.shape
else:
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(estimates)
    distances, indices = nbrs.kneighbors(labels)
    distances = distances[60:-30].flatten()*0.0272
    distances = distances[distances<2.2]
    distances = distances[distances>0.25]

plt.figure()
plt.hist(distances, bins=50, normed=False)
plt.xlabel("Center-To-Wheel Distance (m)")
plt.ylabel("Count")
plt.title("Center-To-Wheel Distance Distribution")
#plt.show()
plt.savefig("results/validation.pdf")

print "Mean:", np.mean(distances)
print "Standard Deviation:", np.std(distances) 
#print "Skewness:", scipy.stats.skew(distances)

if False:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection="3d")
    colors = matplotlib.cm.jet(np.linspace(0,1,labels_.shape[1]))
    for veh, c in zip(range(labels_.shape[1]), colors):
        ax.scatter(range(labels_.shape[0]),
                   labels_[:,veh,1],
                   labels_[:,veh,2],
                   color=c)
    ax.axis("equal")

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1, projection="3d")
    ax1.scatter(labels[:,0], labels[:,1], labels[:,2])
    ax2 = fig.add_subplot(2,1,2, projection="3d")
    ax2.scatter(estimates[:,0], estimates[:,1], estimates[:,2])
    plt.show()
