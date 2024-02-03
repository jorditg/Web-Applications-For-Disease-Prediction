# import the necessary packages
import argparse
import cv2
import math
import ellipsefitter
import numpy as np
from os import listdir
from os.path import isfile, join

def list_files(mypath):
  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
  return onlyfiles
 
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
 
def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt.append((x, y))
		
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function

path = args["path"]
files = list_files(path)
actual_file = 0

image = cv2.imread(join(path, files[actual_file]))
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'e' key is pressed, fit an ellipse with recorsed points
	if key == ord("e"):
	  # draw ellipse
	  x = []
	  y = []
	  for i in range(len(refPt)):
	    x.append(refPt[i][0])
	    y.append(refPt[i][1])
	  x = np.array(x)
	  y = np.array(y)
	  refPt = []
	  ellipse = ellipsefitter.fitEllipse(x,y)
	  center = ellipsefitter.ellipse_center(ellipse)
	  angle = ellipsefitter.ellipse_angle_of_rotation2(ellipse)	   
	  axis = ellipsefitter.ellipse_axis_length(ellipse)
	  startAngle = 0 
	  endAngle = 360
	  color = (0,0,255)
	  thickness = 1
	  center = (int(center[0]),int(center[1]))
	  axis = (int(axis[0]), int(axis[1]))
	  angle_degrees = 180./3.141592*angle
	  axes=(max(axis),min(axis))
	  #print(angle_degrees)
	  cv2.ellipse(image, center, axes, angle_degrees, startAngle, endAngle, color, thickness) 
	  cv2.imshow("image", image)
	  refPt = []
#	  image = clone.copy()	
  # key p saves last point pressed  
	elif key == ord("p"):
	  last = len(refPt) - 1
	  point = (refPt[last][0], refPt[last][1])
	  refPt = []
	elif key == ord("d"):
	  refPt = []
# if 's' pressed, metadata is saved
	elif key == ord("s"):
	  f = open(files[actual_file] + '.txt', 'w')
	  f.write("center0,center1,angle,axis0,axis1,point0,point1\n")
	  f.write(str(center[0]) + "," + str(center[1]) + "," + str(angle) + "," + str(axis[0]) + "," + str(axis[1]) + "," + str(point[0]) + "," + str(point[1]))
	  f.close()
	  refPt = []
# if 'n' pressed, next file is loaded
	elif key == ord("n"):
	  actual_file = actual_file + 1
	  if actual_file == len(files):
	    break
	  image = cv2.imread(join(path, files[actual_file]))
	  clone = image.copy()
	  refPt = []
# if the 'q' key is pressed, break from the loop
	elif key == ord("q"):
		break
 
# close all open windows
cv2.destroyAllWindows()

