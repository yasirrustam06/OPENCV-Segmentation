import cv2
import matplotlib.pyplot as plt
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    ret,Image=cap.read()

    RGB_Image=cv2.cvtColor(Image,cv2.COLOR_BGR2RGB)
    reshaped_Image=RGB_Image.reshape((-1,3))
    Float_Image=np.float32(reshaped_Image)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)


    attempts=10
    k=3

    compactness,labels,center=cv2.kmeans(Float_Image,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center=np.uint8(center)
    res=center[labels.flatten()]
    OutPut_Image=res.reshape((Image.shape))


    cv2.imshow("Original_Image",Image)
    cv2.imshow("OutPut_Image",OutPut_Image)
    cv2.waitKey(1)


