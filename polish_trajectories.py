import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.style.use(["seaborn-deep"])
#plt.style.use("ggplot")
import matplotlib
matplotlib.rc("font", family="FreeSans", size=16)
import sys
import copy
import scipy
from scipy import io

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2) - r
    phi = np.arctan2(y, x)
    return [rho, phi]

def unwrap(wrapped):
    unwrapped = np.zeros_like(wrapped)
    num_veh = wrapped.shape[0]
    duration = wrapped.shape[1]
    for veh in range(num_veh):
        unwrapped[veh,:] = wrapped[veh,:]
        tmp = np.array(wrapped)
        for idx in range(duration-1):
            if tmp[veh, idx+1] - tmp[veh, idx] > np.pi:
                tmp = tmp - 2*np.pi
                unwrapped[veh, idx+1] = tmp[veh, idx+1]
                for t in np.arange(2,30*3):
                    try:
                        if tmp[veh, idx+t] - tmp[veh, idx+1] < -np.pi:
                            tmp[veh, idx+t] += 2*np.pi
                    except:
                        pass
            else:
                unwrapped[veh, idx+1] = tmp[veh, idx+1]
    return unwrapped

SRC = sys.argv[1]

trajectories = np.load(SRC)
detections = np.load(SRC.replace("trajectories", "detections"))
duration = len(detections)
print "Test duration: %d" % duration
num_veh = trajectories.shape[0]
radius = 10.45 * num_veh / (2 * np.pi)
dim = detections[0]
r = dim[1]/(2*np.pi)
R = r + dim[0]

for veh in range(len(trajectories)):
    trajectory = trajectories[veh]
    for num, point in enumerate(trajectory):
        trajectory[num,1:3] = cart2pol(point[1]-R, point[2]-R)
    trajectories[veh] = trajectory

colors = matplotlib.cm.rainbow(np.linspace(1,0,len(trajectories)))
fig1 = plt.figure()
#ax11 = fig1.add_subplot(2,1,1)
#for trajectory, c in zip(trajectories,colors):
#    ax11.plot(trajectory[:,0],trajectory[:,2],color=c)

trajectories_ = []
for veh in range(len(trajectories)):
    trajectory = trajectories[veh][30*2:-30*2,:]

    head = []
    for idx in range(int(trajectory[0,0])):
        head.append([idx, trajectory[0,1], trajectory[0,2], 0])
    if head:
        trajectory = np.insert(trajectory, 0, head, axis=0)
    raw_trajectory = copy.deepcopy(trajectory)
    
    trajectory = []
    trajectory.append(raw_trajectory[0,:])
    for pt1, pt2 in zip(raw_trajectory[0:-1,:], raw_trajectory[1:,:]):
        if pt1[0] == pt2[0]:
            continue
        step = pt2[0] - pt1[0] + 1
        for idx, x, y, v in zip(np.linspace(pt1[0], pt2[0], step),
                                np.linspace(pt1[1], pt2[1], step),
                                np.linspace(pt1[2], pt2[2], step),
                                np.linspace(pt1[3], pt2[3], step)):
            if idx != pt1[0]:
                trajectory.append([idx, x, y, v])
    trajectory = np.array(trajectory)
    
    tail = []
    for idx in range(duration-len(trajectory)):
        tail.append([trajectory[-1,0]+idx+1, trajectory[-1,1],
                     trajectory[-1,2], 0])
    if tail:
        trajectory = np.append(trajectory, tail, axis=0)

    trajectories_.append(trajectory)
trajectories_ = np.asarray(trajectories_)

#ax12 = fig1.add_subplot(2,1,2)
#for trajectory, c in zip(trajectories_,colors):
#    ax12.plot(trajectory[:,0],trajectory[:,2],color=c)

# Revert to old trajectory format
wrapped = []
for trajectory in trajectories_:
    wrapped.append(trajectory[:, 1:3])
wrapped = np.array(wrapped)
unwrapped = copy.deepcopy(wrapped)
unwrapped[:, :, 1] = unwrap(wrapped[:, :, 1])
wrapped[:,:,1] = -wrapped[:,:,1]*radius
unwrapped[:,:,1] = -unwrapped[:,:,1]*radius

scipy.io.savemat(SRC.replace(".npy", "_wrapped_raw.mat"), 
                 mdict={"wrapped": wrapped})
scipy.io.savemat(SRC.replace(".npy", "_unwrapped_raw.mat"), 
                 mdict={"unwrapped": unwrapped})


for veh in range(num_veh):
    unwrapped[veh,:,1] = lowess(unwrapped[veh,:,1], range(duration), 
                                is_sorted=True, frac=0.0125, it=0)[:,1]
    unwrapped[veh,:,0] = lowess(unwrapped[veh,:,0], range(duration), 
                                is_sorted=True, frac=0.0125, it=0)[:,1]
wrapped[:, :, 1] = unwrapped[:, :, 1] % (2*np.pi*radius) 

fig2 = plt.figure(figsize=(21,7))
xmarks = np.arange(0, duration, 300)
xlabels = [str(i) for i in np.arange(0, duration/30, 10)]
ylabels = [str(i) for i in np.arange(0, 360, 30)]

if True:
    ax21 = fig2.add_subplot(1,1,1)
    for trajectory, c in zip(wrapped,colors):
        ax21.scatter(range(len(trajectory[:,1])),
                     trajectory[:,1],
                     color="blue",marker='.',s=0.5)
    ax21.set_xlim([0, duration])
    ax21.set_ylim([np.min(wrapped[:, :, 1]), 
                   np.max(wrapped[:, :, 1])])
    ax21.set_xticks(xmarks)
    ax21.set_xticklabels(xlabels)
    ax21.set_ylabel("Distance (m)")
    ax21.set_xlabel("Time (sec)")
    ax21.set_title("Test %s Trajectories" % SRC[-6:-4])
    ax21.grid(1)

if False:
    ax22 = fig2.add_subplot(2,1,2)
    for trajectory, c in zip(unwrapped,colors):
        ax22.plot(trajectory[:,1],color=c)
    ax22.set_xlim([0, duration])
    ax22.set_ylim([np.min(unwrapped[:, :, 1]), 
                  np.max(unwrapped[:, :, 1])])
    ax22.set_xticks(xmarks)
    ax22.set_xticklabels(xlabels)
    ax22.set_ylabel("Degree (rad)")
    ax22.set_xlabel("Time (sec)")
    ax22.set_title("Deg over Time (Unwrapped)")
#plt.show()
plt.savefig(SRC.replace(".npy", ".png"), bbox_inches="tight")

scipy.io.savemat(SRC.replace(".npy", "_wrapped.mat"), 
                 mdict={"wrapped": wrapped})
scipy.io.savemat(SRC.replace(".npy", "_unwrapped.mat"), 
                 mdict={"unwrapped": unwrapped})
