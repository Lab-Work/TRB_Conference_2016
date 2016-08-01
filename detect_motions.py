# Created by Fangyu Wu
# June 15th, 2016

import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use("seaborn-deep")
import matplotlib
matplotlib.rc("font", family="FreeSans")
import os
from skimage.filters import threshold_otsu
from sklearn.cluster import MiniBatchKMeans, DBSCAN
import scipy
from scipy import io
import sys
import random
import warnings
warnings.filterwarnings("ignore")

def Rec2Ann(x, y):
    rho = y + r
    phi = float(x)/dim[1]*2*np.pi + np.pi/2
    x_ = rho*np.cos(phi) + R
    y_ = rho*np.sin(phi) + R
    return np.array([x_, y_])

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (rho, phi)

def img2ann(img):
    annulus = np.zeros((2*R, 2*R, 3)).astype("uint8")
    for y in range(2*int(R)):
        for x in range(2*int(R)):
            rho, phi = cart2pol(x-R, y-R)
            if r < rho <= R:
                xi = int(dim[1] * phi*rho / (2*np.pi*rho))
                yi = int(rho - r - 1)
                annulus[x, y, :] = img[yi, xi, :]
    return annulus

# Parse the command line argument
SRC = sys.argv[1]
IFPLOT = bool(sys.argv[2])
print IFPLOT

print "Reading %s ..." % SRC
cap = cv2.VideoCapture(SRC)
fourcc = cv2.VideoWriter_fourcc(*'XVID')

ret, rec1 = cap.read()
dim = rec1.shape
r = dim[1]/(2*np.pi)
R = r + dim[0]
if IFPLOT:
    out = cv2.VideoWriter("results/out_"+SRC[-13:-11]+".avi",
                          fourcc,30.0,(dim[1], 2*dim[0]))
    detections = []
    detections.append(dim)
prvs = cv2.cvtColor(rec1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(rec1)
hsv[...,1] = 255

idx = 1
sys.stdout.write("Processing frame %d...\r" % idx)
while(cap.isOpened()):
    ret, rec2 = cap.read()
    if ret == False:
        break
    next = cv2.cvtColor(rec2,cv2.COLOR_BGR2GRAY)
    idx += 1
    sys.stdout.write("Processing frame %d...\r" % idx)
    sys.stdout.flush()

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 
                                        3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    
    loc_mask = np.logical_and(mag > 0.5, ang < 1.1*np.pi)
    loc_mask = np.logical_and(loc_mask, ang > 0.9*np.pi)
    locs = np.column_stack(np.where(loc_mask))
    sample_mask = np.random.choice([False, True], len(locs), p=[0.95, 0.05])
    locs = locs[sample_mask]
    locs = np.array(locs)
    
    if idx == 500 and IFPLOT:
        fig = plt.figure(figsize=(24,8))
        
        rec2 = img2ann(rec2)
        ax1 = fig.add_subplot(1,3,1)
        ax1.imshow(cv2.cvtColor(rec2,cv2.COLOR_BGR2RGB))
        ax1.axis("off")
        #ax1.set_xticklabels([])
        #ax1.set_yticklabels([])
        #ax1.set_title("Original Frame")
        plt.grid(0)
        
        bgr = img2ann(bgr)
        ax2 = fig.add_subplot(1,3,2)
        ax2.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        ax2.axis("off")
        #ax2.set_xticklabels([])
        #ax2.set_yticklabels([])
        #ax2.set_title("Background Subtraction")
        plt.grid(0)

        for num, pixel in enumerate(locs):
            locs[num,:] = Rec2Ann(pixel[1], pixel[0])
        mbk = MiniBatchKMeans(n_clusters=10)
        mbk.fit(locs)
        labels = mbk.labels_.astype(np.int)
        labels_unique = np.unique(labels)
        centers = mbk.cluster_centers_
        colors = matplotlib.cm.jet(np.linspace(0,1,10))
        
        ax3 = fig.add_subplot(1,3,3)
        for lb, c in zip(labels_unique, colors):
            lb_mask = labels == lb
            data = locs[lb_mask]
            ax3.scatter(data[:,0], data[:,1], color=c)
            ax3.plot(centers[lb][0], centers[lb][1], "+k", markersize=25)
        ax3.axis("equal")
        ax3.axis("off")
        #ax3.set_xticklabels([])
        #ax3.set_yticklabels([])
        ax3.set_xlim([0, 2*int(R)])
        ax3.set_ylim([0, 2*int(R)])
        ax3.invert_xaxis()
        ax3.invert_yaxis()
        #ax3.set_title("K-Means Clustering")
        
        plt.savefig("results/optflowkmeans.pdf", bbox_inches="tight")
        break
    elif not IFPLOT:
        for loc in locs:
            cv2.circle(rec2, (int(loc[1]), int(loc[0])), 
                       1, (0, 255, 0), -1)
        rec = np.vstack((rec2, bgr))
        out.write(rec)
        cv2.imshow('rectangle',rec)
        
        locs = np.array([[item[0], item[1], mag[item[0], item[1]]] 
                         for item in locs])
        detections.append(locs)
        np.save("results/detections_"+SRC[-13:-11], detections)

    prvs = next
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows() 

print "\nDone."
