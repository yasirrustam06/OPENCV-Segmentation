import cv2
import numpy as np
import matplotlib.pyplot as plt


Image=cv2.imread("kmeans.webp")

rgb_img = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)

light_blue = (90, 70, 50)
dark_blue = (128, 255, 255)
# You can use the following values for green
# light_green = (40, 40, 40)
# dark_greek = (70, 255, 255)
mask = cv2.inRange(hsv_img, light_blue, dark_blue)
result = cv2.bitwise_and(Image, Image, mask=mask)

cv2.imshow("Original_Image",Image)
cv2.imshow("OutPut_Image",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
