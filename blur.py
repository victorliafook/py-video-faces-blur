import pickle
import json
import numpy as np
import cv2
from collections import OrderedDict 
from moviepy.editor import *
#from moviepy.video.tools.tracking import manual_tracking
from scipy.interpolate import interp1d


# Need to improve this function: We need to interpolate the position of each person!

def to_fxfy(txy_list, **kwargs):
    """ Transforms a list [ (ti, (xi,yi)) ] into 2 functions (fx,fy)
        where fx : t -> x(t)  and  fy : t -> y(t).
        If the time t is out of the bounds of the tracking time interval
        fx and fy return the position of the object at the start or at
        the end of the tracking time interval.
        Keywords can be passed to decide the kind of interpolation,
        see the doc of ``scipy.interpolate.interp1d``."""                                                                        
        
    tt, pointsXXYY = zip(*txy_list.items())
    
    xx, yy = zip(*pointsXXYY)
    xx1, xx2 = zip(*xx)
    yy1, yy2 = zip(*yy)
    
    interp_x1 = interp1d(tt, xx1, **kwargs)
    interp_y1 = interp1d(tt, yy1, **kwargs)
    fx1 = lambda t: xx1[0] if (t <= tt[0]) else ( xx1[-1] if t >= tt[-1]
                                          else ( interp_x1(t) ) )
    fy1 = lambda t: yy1[0] if (t <= tt[0]) else ( yy1[-1] if t >= tt[-1]
                                          else ( interp_y1(t) ) )
    
    interp_x2 = interp1d(tt, xx2, **kwargs)
    interp_y2 = interp1d(tt, yy2, **kwargs)
    fx2 = lambda t: xx2[0] if (t <= tt[0]) else ( xx2[-1] if t >= tt[-1]
                                          else ( interp_x2(t) ) )
    fy2 = lambda t: yy2[0] if (t <= tt[0]) else ( yy2[-1] if t >= tt[-1]
                                          else ( interp_y2(t) ) )
                                          
    return fx1,fy1,fx2,fy2
    
def headblur(clip,fx1,fy1,fx2,fy2,r_zone,r_blur=None):
    """
    Returns a filter that will blurr a moving part (a head ?) of
    the frames. The position of the blur at time t is
    defined by (fx(t), fy(t)), the radius of the blurring
    by ``r_zone`` and the intensity of the blurring by ``r_blur``.
    Requires OpenCV for the circling and the blurring.
    Automatically deals with the case where part of the image goes
    offscreen.
    """
    
    if r_blur is None: r_blur = 2*r_zone/3
    
    def fl(gf,t):
        
        im = gf(t)
        h,w,d = im.shape
        x1,x2 = int(fx1(t)), int(fx2(t))
        y1,y2 = int(fy1(t)), int(fy2(t))
        
        x1,x2 = max(0,x1),min(x2,w)
        y1,y2 = max(0,y1),min(y2,h)
        
        # Coordinates are from top left 0,0
        region_size = y2-y1,x2-x1
        #print region_size
        
        mask = np.zeros(region_size).astype('uint8')
        region_size = np.array(region_size)
        r_blur = 2*region_size/3
        r_blur = tuple(r_blur)
        region_size = tuple(region_size)
        
        #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]]) 
        #cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) 
        
        #cv2.circle(mask, (region_size[0]/2,region_size[1]/2), region_size[0]/2, 255, -1, lineType=cv2.CV_AA)
        #cv2.rectangle(mask, (x1,y1), (x2,y2), 255, -1, lineType=cv2.CV_AA)
                               
        mask = np.dstack(3*[(1.0/255)*mask])
        
        orig = im[y1:y2, x1:x2]
        blurred = cv2.blur(orig,(region_size[0]/2, region_size[0]/2))
        im[y1:y2, x1:x2] = mask*blurred + (1-mask)*orig
        
        return im
    
    return clip.fl(fl)

inputData = []

with open("input/childrenFaces.json",'r') as f:
    inputData = json.load(f)
    
faces = inputData["Persons"]
facesLocalizer = {}
sortedFacesLocalizer = {}

frameWidth = inputData["VideoMetadata"]["FrameWidth"]
frameHeight = inputData["VideoMetadata"]["FrameHeight"]
duration = inputData["VideoMetadata"]["DurationMillis"]

widerRatio = 0.05

for elem in faces:
    
    
    try:
        if elem["Person"]["Face"] != None:
            if facesLocalizer.get(elem["Person"]["Index"]) == None:
                facesLocalizer[elem["Person"]["Index"]] = OrderedDict()
            
            face = elem["Person"]["Face"]
            faceXX = ( face["BoundingBox"]["Left"] * frameWidth * (1-widerRatio), (face["BoundingBox"]["Left"] + face["BoundingBox"]["Width"]) * frameWidth * (1+widerRatio))
            faceYY = ( face["BoundingBox"]["Top"] * frameHeight * (1-widerRatio), (face["BoundingBox"]["Top"] + face["BoundingBox"]["Height"]) * frameHeight * (1+widerRatio))
            
            faceLocal = (faceXX, faceYY)
            #if facesLocalizer.get(elem["Person"]["Index"]).get(elem["Timestamp"]) == None :
            #    facesLocalizer[elem["Person"]["Index"]][elem["Timestamp"]] = []
            facesLocalizer[elem["Person"]["Index"]][elem["Timestamp"]/1000] = faceLocal
    except KeyError:
        pass


print(facesLocalizer)
# for key, value in facesLocalizer.iteritems():
#     print "\n\n"
#     print "#######################\n"
#     for ikey in sorted(facesLocalizer[key].iterkeys()):
#         if sortedFacesLocalizer.get(key) == None:
#             sortedFacesLocalizer[key] = {}
#         sortedFacesLocalizer[key][ikey] = facesLocalizer[key][ikey]
#         print ikey
#         print sortedFacesLocalizer[key][ikey]

# LOAD THE ORIGINAL CLIP 
clip = VideoFileClip("media/childrenFaces.mp4")

# MANUAL TRACKING OF THE HEAD

# the three next lines are for the manual tracking and its saving
# to a file, it must be commented once the tracking has been done
# (after the first run of the script for instance).
# Note that we save the list (ti,xi,yi), not the functions fx and fy
# (that we will need) because they have dependencies.

#txy, (fx,fy) = manual_tracking(clip, fps=6)
#with open("../../chaplin_txy.dat",'w+') as f:
#    pickle.dump(txy)

# VICTOR: Returns a list [(t1,x1,y1),(t2,x2,y2) etc... ] if there is one
# object per frame, else returns a list whose elements are of the 
# form (ti, [(xi1,yi1), (xi2,yi2), ...] )


# IF THE MANUAL TRACKING HAS BEEN PREVIOUSLY DONE,
# LOAD THE TRACKING DATA AND CONVERT IT TO FUNCTIONS x(t),fy(t)

#with open("../../chaplin_txy.dat",'r') as f:
    #fx,fy = to_fxfy( pickle.load(f) )
for key in sorted(facesLocalizer.iterkeys()):
    print "\n\n#######################\n"
    print facesLocalizer[key]
    # print "\n"
    fx1,fy1,fx2,fy2 = to_fxfy( facesLocalizer[key] , kind='linear')
    
    # https://docs.aws.amazon.com/rekognition/latest/dg/API_BoundingBox.html
    # CONSTRUCT THE POINTS COORDINATES ACCORDING TO EACH FACES BOUNDING BOX
    
    # BLUR CHAPLIN'S HEAD IN THE CLIP
    
    clip = clip.fx( headblur, fx1,fy1,fx2,fy2, 1000)
    
    
    # Generate the text, put in on a grey background
    
    #txt = TextClip("Hey you ! \n You're blurry!", color='grey70',
                #   size = clip.size, bg_color='grey20',
                #   font = "Century-Schoolbook-Italic", fontsize=40)
                   
                   
    # Concatenate the Chaplin clip with the text clip, add audio
    
    #final = concatenate_videoclips([clip_blurred,txt.set_duration(3)]).\
              #set_audio(clip.audio)
    
    # We write the result to a file. Here we raise the bitrate so that
    # the final video is not too ugly.

clip.write_videofile('output/blurredChildrenMovieWidertest.mp4')

