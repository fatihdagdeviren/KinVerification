#Import required modules
import cv2
import dlib
from skimage.feature import local_binary_pattern
import numpy as np
from scipy.stats import itemfreq 
from sklearn.preprocessing import normalize 

METHOD = 'uniform'
radius = 1
n_points = 8 * radius

def get_landmarks(path,detector,predictor,patchSize,eps=1e-7):
    im = cv2.imread(path)
    image = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    detections = detector(image, 1)
    patchSize = int(patchSize/2)
    landmarks=[]    
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            
        for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.            
            roi = image[y-patchSize:y+patchSize+1,x-patchSize:x+patchSize+1]           
            lbp = local_binary_pattern(roi, n_points, radius, METHOD)           
            (hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, n_points + 3),
			range=(0, n_points + 2))
            hist = hist.astype("float")
            hist /= (hist.sum() + eps)
            landmarks.append([(x,y),roi,hist])

        

    if len(detections) > 0:
        return landmarks
    else: #If no faces are detected, return error message to other function to handle
        landmarks = "error"
        return landmarks