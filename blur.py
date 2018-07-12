import json
import argparse
import numpy as np
import cv2
import blur_utils as blur
from collections import OrderedDict 
from moviepy.editor import *
#from moviepy.video.tools.tracking import manual_tracking


# Need to improve this function: We need to interpolate the position of each person!


    
# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
	
args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# LOAD THE ORIGINAL CLIP 
clip = VideoFileClip(args["input"])


clip = clip.fx( blur.headblur, net, args["confidence"])
                   
                   
clip.write_videofile(args["output"])

