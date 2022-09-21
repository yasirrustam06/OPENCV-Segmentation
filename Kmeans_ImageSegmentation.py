import cv2
import matplotlib as plt
import numpy as np

Image=cv2.imread('kmeans.jpg')
Rgb_Image=cv2.cvtColor(Image,cv2.COLOR_BGR2RGB)
reshaped_Image=Rgb_Image.reshape((-1,3))

Float_Image=np.float32(reshaped_Image)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 4
attempts=10

ret,label,center=cv2.kmeans(Float_Image,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((Image.shape))

cv2.imshow("original_Image",Image)
cv2.imshow("OutPut_Image",result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

