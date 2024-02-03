# import the necessary packages
import argparse
import cv2
import shutil
from os import listdir
from os.path import isfile, join

def list_files(mypath):
  onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
  return onlyfiles
  
		
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
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("g"):
	  shutil.move(join(path, files[actual_file]), './good/' + files[actual_file]) 
	  actual_file = actual_file + 1
	  if actual_file == len(files):
	    break
	  image = cv2.imread(join(path, files[actual_file]))
	  clone = image.copy()
	  refPt = []

	elif key == ord("b"):
	  shutil.move(join(path, files[actual_file]), './bad/' + files[actual_file]) 
	  actual_file = actual_file + 1
	  if actual_file == len(files):
	    break
	  image = cv2.imread(join(path, files[actual_file]))
	  clone = image.copy()
	  refPt = []

	elif key == ord("q"):
		break
 
# close all open windows
cv2.destroyAllWindows()

