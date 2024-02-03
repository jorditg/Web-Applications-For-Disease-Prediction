#!bin/bash
# github does not allow big files
# in order to load the model we have splitted it
# this script allows its reconstruction in one file 

cat checkpoint_part_* > checkpoint_0_20_9_0.953.pth.tar


