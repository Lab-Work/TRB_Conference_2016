## Source Code for the Paper *Fuel Consumption in Oscillatory Traffic: Experimental Results*
#### Prepared for TRB Conference 2016
#### Written by Fangyu Wu and Raphael Stern

The source runs in Python and has the following dependencies:
+ Numpy
+ Matplotlib
+ OpenCV
+ Scikit-Learn

Please install the required libraries before using the scripts. To run the code, please do:

`python extract_dynamics.py (number of vehicles) (source) (destination)`

For example, to process test A video and save the output to the current directory, one may:

`python extract_dynamics.py 10 ../test_A/edited_video.mp4 ./`

If you spot bugs in the script or encountered any issues, please kindly contact the author Fangyu Wu (fwu10@illinois.edu). We will address the problems as soon as possible. 
