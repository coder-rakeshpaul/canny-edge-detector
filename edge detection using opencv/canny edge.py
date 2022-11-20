
import cv2

img = cv2.imread('image.jpg')
edges = cv2.Canny(img,100,200) # 100 is the lower threshold and 200 is the higher threshold 

cv2.imshow(edges)