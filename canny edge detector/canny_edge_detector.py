

import cv2
import numpy as np
from numpy import pi


original_image = cv2.imread('chad.jpg')
print(original_image.shape)


'''    Noise reduction;
    Gradient calculation;
    Non-maximum suppression;
    Double threshold;
    Edge Tracking by Hysteresis.
'''
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g
'''

blur_image= cv2.filter2D(original_image, ddepth=-1,kernel= gaussian_kernel(9,1))
cv2_imshow(blur_image)
'''
def sobel():
  sobely=np.array([[1,2,1],
                  [0,0,0],
                  [-1,-2,-1]])
  
  sobelx=np.array([[1,0,-1],
                  [2,0,-2],
                  [1,0,-1]])
  
  img_sobel_x= cv2.filter2D(original_image, ddepth=-1,kernel=sobelx )
  img_sobel_y= cv2.filter2D(original_image, ddepth=-1,kernel=sobely )

  img_sobel= np.hypot(img_sobel_x,img_sobel_y)
  img_sobel = img_sobel / img_sobel.max() * 255

  theta = np.arctan2(img_sobel_y, img_sobel_x)

 
sobel()