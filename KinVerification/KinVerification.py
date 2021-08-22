#Import required modules
import cv2
import dlib
import FacialLandmark as fl
import os
import numpy as np

landmarks_children = []
landmarks_parents = []

def main(pathForChild):
    landmarks_children.clear()
    landmarks_parents.clear()
    #roi = gray[hei-yeniPatchSize:hei+yeniPatchSize+1,wi-yeniPatchSize:wi+yeniPatchSize+1] 
    patchSize = 8
    #Set up some required objects
    included_extenstions = ['jpg', 'bmp', 'png']
    young_ParentPath = "../KinFace_V2/02"
    old_ParentPath = "../KinFace_V2/03"
    children_Path = "../KinFace_V2/01"

    file_names_children = [children_Path+"/"+fn for fn in os.listdir(children_Path)
                  if any(fn.endswith(ext) for ext in included_extenstions)]
    file_names_oldParent = [old_ParentPath+"/"+fn for fn in os.listdir(old_ParentPath)
                  if any(fn.endswith(ext) for ext in included_extenstions)]
    file_names_youngParent = [young_ParentPath+"/"+fn for fn in os.listdir(young_ParentPath)
                  if any(fn.endswith(ext) for ext in included_extenstions)]
    
    file_names_oldParent = sorted(file_names_oldParent,key=lambda x: int(x[17:-4]))
    file_names_youngParent = sorted(file_names_youngParent,key=lambda x: int(x[17:-4])) 

    file_names_children = [x for x in file_names_children if x==pathForChild][0]
    file_names_oldParent = file_names_oldParent[120:]
    file_names_youngParent = file_names_youngParent[120:]


    detector = dlib.get_frontal_face_detector() 
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

    sift = cv2.xfeatures2d.SIFT_create()
 
    children_Landmark = fl.get_landmarks(file_names_children,detector,predictor,patchSize)
    landmarks_children.append(children_Landmark)
    for x in range(0,len(file_names_youngParent)):        
        # landmarktaki her bir alan noktanin hog'unu hesaplayacagim.      
        oldParent = fl.get_landmarks(file_names_oldParent[x],detector,predictor,patchSize)
        youngParent = fl.get_landmarks(file_names_youngParent[x],detector,predictor,patchSize)
        if oldParent!="error" and youngParent!="error":
            landmarks_parents.append([youngParent,oldParent,file_names_youngParent[x],file_names_oldParent[x]])
   
    #return landmarks_children,landmarks_parents


def ekrandaGoster(pathForChild,enYakinParentlar):
    child = landmarks_children[0]
    for enYakinParent in enYakinParentlar:        
        parent = landmarks_parents[enYakinParent[0]]
        yakinlikDerecesi = enYakinParent[1].fitness.values[1]
        imageChild = cv2.imread(pathForChild)
        path_imageParentYoung = parent[2]
        path_imageParentOld = parent[3]
        imageParentYoung = cv2.imread(path_imageParentYoung)
        imageParentOld = cv2.imread(path_imageParentOld)
        #for x in range(0,len(child)):        
            #cv2.putText(imageChild,"+",(int(child[x][0][0]),int(child[x][0][1])),cv2.FONT_HERSHEY_SIMPLEX, 0.2, 255)
    
        #for m in range(0,len(enYakinParent[1])):
            #deger = enYakinParent[1][m]
            #if deger == 0: #old
                #cv2.putText(imageParentOld,"+",(int(parent[1][m][0][0]),int(parent[1][m][0][1])),cv2.FONT_HERSHEY_SIMPLEX, 0.2, 255)
            #else: # young
                #cv2.putText(imageParentYoung,"+",(int(parent[0][m][0][0]),int(parent[0][m][0][1])),cv2.FONT_HERSHEY_SIMPLEX, 0.2, 255)
        print("En Yakın Parent: {}".format(path_imageParentOld))
        #vis = np.concatenate((imageParentYoung, imageParentOld), axis=0)
        #cv2.imshow("Child({})".format(pathForChild),imageChild)
        #cv2.imshow("YPP-({})".format(path_imageParentYoung),imageParentYoung)
        #cv2.imshow("OPP-({})".format(path_imageParentOld),imageParentOld)
    #cv2.waitKey(0)
    
    


