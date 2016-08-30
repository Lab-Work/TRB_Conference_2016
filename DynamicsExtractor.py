# Authors: Fangyu Wu (fwu10@illinois.edu)
# Date: June 15th, 2016
# Script to extract displacements, velocities, and accelerations from panoramic videos

import numpy as np
import cv2
import matplotlib
matplotlib.rc("font", family="FreeSans")
import matplotlib.pyplot as plt
plt.style.use("seaborn-white")
from sklearn.cluster import MiniBatchKMeans, KMeans, DBSCAN
import scipy
from scipy import io as sio
from scipy.interpolate import interp1d
import sys
import os
import copy
import warnings
warnings.filterwarnings("ignore") #To scilence warnings from MiniBatchKMeans

class DynamicsExtractor:
    # Initialize the class.
    def __init__(self, SRC, order):
        self.SRC = SRC # Source directory
        cap = cv2.VideoCapture(self.SRC+"edited_video.mp4") # Read in source video
        self.NVEH = len(order) # Number of vehicles in the test
        self.order = order # The order in which vehicles were placed in the first frame
        self.dim = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))] # Size of video frame
        self.r = self.dim[1]/(2*np.pi) # Polar layout inner radius
        self.R = self.r + self.dim[0] # Polar layout outer radius
        self.duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total number of frames
        cap.release()
        self.fps = 30.0

    # Transform image layout to polar layout.
    def img2pol(self, xi, yi):
        rho = float(yi) + self.r
        phi = float(xi)/self.dim[1]*2*np.pi + np.pi/2
        yp = rho*np.cos(phi) + self.R
        xp = rho*np.sin(phi) + self.R
        return [round(xp), round(yp)]

    # Recover image layout from polar layout.
    def pol2img(self, xp, yp):
        rho = np.sqrt((xp-self.R)**2 + (yp-self.R)**2)
        phi = np.arctan2(xp-self.R, yp-self.R) - np.pi/2
        xi = self.dim[1]*phi*rho / (2*np.pi*rho)
        if xi < 0:
            xi += self.dim[1]
        yi = rho - self.r
        return [round(xi), round(yi)]

    # Unwrap vehicle trajectory.
    # This step is required for interpolation, smoothing and differentiation.
    def unwrap(self, wrapped):
        duration = len(wrapped)
        unwrapped = copy.deepcopy(wrapped)
        tmp = np.array(wrapped)
        for idx in range(duration-1):
            if tmp[idx+1] - tmp[idx] > 3840/2:
                tmp = tmp - 3840
                unwrapped[idx+1] = tmp[idx+1]
                for t in np.arange(2,30*3):
                    try:
                        if tmp[idx+t] - tmp[idx+1] < -3840/2:
                            tmp[idx+t] += 3840
                    except:
                        pass
            else:
                unwrapped[idx+1] = tmp[idx+1]
        return unwrapped

    # Find the centers of every vehicle in every frame of the source video 
    # by dense optical flow and minibatch KMeans clustering.
    def find_centers(self):
        sys.stdout.write("Reading %sedited_video.mp4...\n" % self.SRC)
        cap = cv2.VideoCapture(self.SRC+"edited_video.mp4") # Read in source video
        ret, frame = cap.read() # Capture the first frame
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale image
        frame_num = 1
        sys.stdout.write("Total number of frames to process: %d\n" % self.duration)
        sys.stdout.write("Processing frame %d...\r" % frame_num)
        centers_cloud = []
        while(1):
            ret, frame = cap.read() # Capture the next frame
            # Break out of the loop when reaching the end of the video file.
            if ret == False:
                break
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale image.
            frame_num += 1
            sys.stdout.write("Processing frame %d...\r" % frame_num)
            sys.stdout.flush()
            # Apply the dense optical flow algorithm proposed by Farneback.
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 
                                                3, 15, 3, 5, 1.2, 0)
            # Convert optical flow output from Cartesian to polat coordinates.
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # Remove small motion and motion with wrong directions.
            mask = np.logical_and(mag > 0.5, ang < 1.1*np.pi)
            mask = np.logical_and(mask, ang > 0.9*np.pi)
            # Find the image coordinates of the foreground pixels.
            fg_pixels = np.column_stack(np.where(mask))
            # Change foreground pixels coordinates to polar layout
            # to deal with the periodicity in the data.
            for num, pixel in enumerate(fg_pixels):
                xi = pixel[1]
                yi = pixel[0]
                fg_pixels[num, 0:2] = self.img2pol(xi, yi)
            # Apply minibatch kmeans to find centers of vehicles in the exp.
            # Since foreground batch fg_pixels may be empty due to small motion, 
            # a conditioning statement is added to the beginning. Note that 
            # fg_pixels with length less than NVEH will result in errors.
            if len(fg_pixels) > 100:
                mbk = MiniBatchKMeans(n_clusters=self.NVEH,batch_size=500)
                mbk.fit(fg_pixels)
                centers = mbk.cluster_centers_
                # Recover foreground pixels image layout from polar layout.
                for num, pixel in enumerate(centers):
                    xp = pixel[0]
                    yp = pixel[1]
                    centers[num, 0:2] = self.pol2img(xp, yp)
                # Preppend timestamp.
                centers = [[frame_num, center[0], center[1]] for center in centers]
                centers_cloud.append(centers)
                # Uncomment to save the progress
                np.save(self.SRC+"centers_cloud.npy", centers_cloud) # Cache raw dynamics.
                # Draw text on the extracted camera frame to visually verify the 
                # the outcomes.
                if False:
                    for num, center in enumerate(centers):
                        num_ = num + 1
                        cv2.putText(frame, str(num_), 
                                    (int(center[1]), int(center[2])), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 
                                    1, cv2.LINE_AA)
                    cv2.imshow("Centers", frame)
            prvs = next
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        sys.stdout.write("\nDone.\n")

    # Find the displacement timeseries of every vehicle in the test using
    # DBSCAN, linear interpolation, and robust LOESS smoother developed by 
    # Maria Laura Delle Monache.
    def find_displacements(self):
        sys.stdout.write("Reading %scenters_cloud.npy...\n" % self.SRC)
        centers_cloud= np.load(self.SRC+"centers_cloud.npy") # Store as numpy array
        print "Size of centers cloud is", centers_cloud.shape
        # Timestamp is stretched by a factor of 2.5 to facilitate clustering.
        centers_cloud = [[2.5*centers_cloud[t,v,0], 
                        centers_cloud[t,v,1], 
                        centers_cloud[t,v,2]] 
                        for t in range(centers_cloud.shape[0]) 
                        for v in range(centers_cloud.shape[1])]
        centers_cloud = np.asarray(centers_cloud)
        # Convert to polar layout
        for t in range(centers_cloud.shape[0]):
            centers_cloud[t,1:] = self.img2pol(centers_cloud[t,1], centers_cloud[t,2])
        # Use DBSCAN to reconstruct displacements
        # Parameters eps and min_samples may need to be tuned case by case.
        dbscan = DBSCAN(eps=36, min_samples=1)  
        dbscan.fit(centers_cloud)
        labels = dbscan.labels_
        labels_unique = np.unique(labels)
        # Remove small clusters (noise)
        displacements = []
        for label in labels_unique:
            mask = labels == label
            cluster = centers_cloud[mask]
            if len(cluster) > 30*120:
                print "Discovered a trajectory of shape", cluster.shape
                # Recover image layout from polar layout
                for idx in range(len(cluster)):
                    cluster[idx, 1:] = self.pol2img(cluster[idx,1], cluster[idx,2])
                displacements.append(cluster)
        displacements = np.asarray(displacements) # Store as numpy array
        for idx, displacement in enumerate(displacements):
            displacements[idx][:,0] /= 2.5
        print "Found %d displacements." % displacements.shape[0]
        if displacements.shape[0] != self.NVEH:
            print "Warning: there should be %d trajectories." % self.NVEH
            print "Try to adjust DBSCAN parameters."
        # Unwrap the displacements timeseries.
        for veh, displacement in enumerate(displacements):
            displacements[veh][:,1] = self.unwrap(displacement[:,1]) 
        displacements_tmp = []
        for veh in range(len(displacements)):
            displacement = displacements[veh]
            # Remove redunant reading.
            displacement = displacement[np.unique(displacement[:,0], return_index=True)[1]]
            # Extrapolate the data with the first reading.
            if displacement[0,0] != 1:
                head = [[0, displacement[0,1], displacement[0,2]]]
                displacement = np.insert(displacement, 0, head, axis=0)
            # Extrapolate the data with the last reading.
            if displacement[-1,0] != self.duration:
                tail = [[self.duration-1, displacement[-1,1], displacement[-1,2]]]
                displacement = np.insert(displacement, -1, tail, axis=0)
            displacement = displacement[displacement[:,0].argsort()]
            x = displacement[:,0]
            y = np.transpose(displacement[:,1:])
            # Fill in missing data in between with linear interpolation.
            f = interp1d(x, y, kind="linear", axis=1)
            X = np.arange(self.duration)
            displacement = np.transpose(f(X))
            displacements_tmp.append(displacement)
        displacements = np.asarray(displacements_tmp)
        # Sort the trajectories by initial location.
        sort_idx = np.argsort(displacements[:,0,0])
        displacements_sort = displacements[sort_idx]
        # Align trajectories data with vehicle ID and change coordinates system
        # to a more convenient form.
        for veh in range(self.NVEH):
            # Invert the X axis to make displacement positive.
            displacements[self.order[veh]-1,:,0] = -displacements_sort[veh,:,0]
            # Change the origin from the upper left corner to lower right corner.
            displacements[self.order[veh]-1,:,1] = self.dim[0] - displacements_sort[veh,:,1]
        displacements *=  10.45*9./3840. # Convert the uinit from pixel to meter.
        np.save(self.SRC+"displacements.npy", displacements)
        scipy.io.savemat(self.SRC+"displacements.mat", mdict={"displacements": displacements})

    # Find the velocity timeseries of every vehicle in the test using
    # regularized differentiation developed by Maria Laura Delle Monache.
    # (Implemented in Matlab. Need to be ported into Python at some point.)
    def find_velocities(self):
        pass

    # Find the acceleration timeseries of every vehicle in the test using
    # regularized differentiation developed by Maria Laura Delle Monache.
    # (Implemented in Matlab. Need to be ported into Python at some point.)
    def find_accelerations(self):
        pass

    # Qualitatively validate the estimated displacements by overlaying
    # the vehicle ID at the estimated displacements on top of the video.
    # By default this checks the smoothed displacements.
    def inspect_displacements(self, check_smoothed=True):
        print "Overlaying vehicle ID on video..."
        dynamics = sio.loadmat(self.SRC+"dynamics.mat")
        cap = cv2.VideoCapture(self.SRC+"edited_video.mp4") # Read in source video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if check_smoothed:
            displacements = dynamics['displacements_smoothed']
            out = cv2.VideoWriter(self.SRC+"overlay_smoothed.avi",fourcc, 30.0, 
                                (self.dim[1],self.dim[0]))
        else:
            displacements = np.load(self.SRC+"displacements.npy")
            out = cv2.VideoWriter(self.SRC+"overlay_unsmoothed.avi",fourcc, 30.0, 
                                (self.dim[1],self.dim[0]))
        time = 0
        while(1):
            ret, frame = cap.read() # Capture the next frame
            # Break out of the loop when reaching the end of the video file.
            if ret == False:
                break
            for id in range(self.NVEH):
                veh_ID = str(id + 1)
                center = displacements[id, time, :]
                x = int((-center[0] % (10.45*9.)) * 3840./(10.45*9.))
                y = int((-center[1] + self.dim[0]*10.45*9./3840.) * 3840./(10.45*9.))
                cv2.putText(frame, veh_ID, (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 
                            2, cv2.LINE_AA)
            #cv2.imshow("Inspect Displacements", frame)
            out.write(frame)
            time += 1 
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()
        print "Done."

    # Visualize one component of the final displacements, velocities and 
    # accelerations. Default to horizontal component. 
    def visualize(self, horizontal=True, save=False):
        print "Plotting " + self.SRC[:-1] + " dynamics..."
        if horizontal:
            idx = 0 # Index 0 for horizontal component
        else:
            idx = 1 # Index 1 for vertical component
        # Load data from .mat file.
        dynamics = sio.loadmat(self.SRC+"dynamics.mat")
        displacements_smoothed = dynamics['displacements_smoothed']
        velocities_regularized = dynamics['velocities_regularized']
        acceleration_regularized = dynamics['accelerations_regularized']
        colors = matplotlib.cm.plasma(np.linspace(1,0,self.NVEH))
        fig = plt.figure(figsize=(15,15))
        # Visualize smoothed displacements
        ax1 = fig.add_subplot(3,1,1)
        for veh, color in zip(range(self.NVEH), colors):
            veh_ID = str(veh+1)
            ax1.plot(np.arange(self.duration)/self.fps, 
                     displacements_smoothed[veh,:,idx], label=veh_ID)
        ax1.set_xlabel("Time (sec)")
        ax1.set_ylabel("Smoothed Displacements (m)")
        #ax1.legend()
        ax1.grid(1)
        ax1.set_title("Test %s Dynamics" % self.SRC[-2])
        # Visualize regularized velocities
        ax2 = fig.add_subplot(3,1,2)
        for veh, color in zip(range(self.NVEH), colors):
            veh_ID = str(veh+1)
            ax2.plot(np.arange(self.duration)/self.fps, 
                     velocities_regularized[veh,:,idx], label=veh_ID)
        ax2.set_xlabel("Time (sec)")
        ax2.set_ylabel("Regularized Velocities (m/s)")
        #ax2.legend()
        ax2.grid(1)
        # Visualize regularized accelerations
        ax3 = fig.add_subplot(3,1,3)
        for veh, color in zip(range(self.NVEH), colors):
            veh_ID = str(veh+1)
            ax3.plot(np.arange(self.duration)/self.fps, 
                     acceleration_regularized[veh,:,idx], label=veh_ID)
        ax3.set_xlabel("Time (sec)")
        ax3.set_ylabel("Regularized Accelerations (m/s2)")
        #ax3.legend()
        ax3.grid(1)
        # Choose to save or display the plot.
        if save:
            plt.savefig(self.SRC+"dyanmics.pdf")
        else:
            plt.show()
        print "Done."

if __name__=="__main__":
    print "Initiating dynamics extractor..."
    tests = ["test_A/", "test_B/", "test_C/", "test_D/"]
    # The initial order in which vehicles were placed in the tests.
    orders = [[8, 7, 6, 5, 4, 3, 2, 1, 10, 9],  # Test A
              [2, 1, 9, 8, 7, 6, 5, 4, 3],      # Test B
              [8, 7, 6, 5, 4, 3, 2, 1, 10, 9],  # Test C
              [6, 5, 4, 3, 2, 1, 9, 8, 7]]      # Test D
    for test_X, order_X in zip(tests, orders):
        extractor = DynamicsExtractor(test_X, order_X)
        '''////////////////////////////////////////////////////////////////
        STEP 1: Extracting Unsmoothed Displacements
        ////////////////////////////////////////////////////////////////'''
        #extractor.find_centers()
        #extractor.find_displacements()
        '''////////////////////////////////////////////////////////////////
        STEP 2: Smoothing and Differentiation
        Not implemented in Python. Execute the DynamicsExtractor.m 
        script instead.
        ////////////////////////////////////////////////////////////////'''
        #extractor.find_velocities()
        #extractor.find_accelerations()
        '''////////////////////////////////////////////////////////////////
        STEP 3: Visualization
        Executable after STEP 2.
        ////////////////////////////////////////////////////////////////'''
        extractor.visualize(save=True)
        extractor.inspect_displacements()
