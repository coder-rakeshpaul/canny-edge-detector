
import cv2
import numpy as np

def gaussian_blur(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    blur_img = cv2.filter2D(original_image, ddepth=-1,kernel=g )
    return blur_img

def sobel(blur_img):
  sobely=np.array([[1,2,1],
                  [0,0,0],
                  [-1,-2,-1]])
  
  sobelx=np.array([[1,0,-1],
                  [2,0,-2],
                  [1,0,-1]])
  
  img_sobel_x= cv2.filter2D(blur_img, ddepth=-1,kernel=sobelx )
  img_sobel_y= cv2.filter2D(blur_img, ddepth=-1,kernel=sobely )

  img_sobel= np.hypot(img_sobel_x,img_sobel_y)
  img_sobel = img_sobel / img_sobel.max() * 255
  theta = np.arctan2(img_sobel_y, img_sobel_x)

  return (img_sobel, theta)
 
def nom_max_supression(img_sobel,theta):

  m,n = img_sobel.shape
  z = np.zeros((m,n))
  angle = theta * 180 / np.pi
  angle[angle < 0] += 180

  q=255
  r=255

  for i in range(m-1):
    for j in range(n-1):

      # 0
      if (0 <= angle[i,j] <22.5 or 157.5 < angle[i,j] <=180 ):
        q = img_sobel[i, j+1]
        r = img_sobel[i, j-1]

       # 45 
      elif (22.5 <= angle[i,j] < 67.5 ):  
        q = img_sobel[i+1, j-1]
        r = img_sobel[i-1, j+1]

        # 90
      elif (67.5 <= angle[i,j] < 112.5 ):  
        q = img_sobel[i+1, j]
        r = img_sobel[i-1, j]

        # 135

      elif (112.5 <= angle[i,j] < 157.5 ):  
        q = img_sobel[i-1, j-1]
        r = img_sobel[i+1, j+1]   

      if ( img_sobel[i,j] >= q ) and img_sobel[i,j] >= r:
        z[i,j] = img_sobel[i,j]

      else:
        z[i,j]=0     

  return z

def double_threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio;
    lowThreshold = highThreshold * lowThresholdRatio;
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(50)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img    


original_image = cv2.imread('image.png')
original_image =  cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)    
           
blur_img =gaussian_blur(5)

sobel , theta = sobel(blur_img)
nms = nom_max_supression(sobel , theta)

img , weak , strong =  double_threshold(nms)
output = hysteresis(img , weak , strong)
cv2.imshow(output)
