import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
def rotate_image(image, angle): ## rotate left degrees
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result
img = cv2.imread('man.png', cv2.IMREAD_GRAYSCALE)
img_rotate_30 = rotate_image(img,30)


l1 = cv2.imread('man.png',cv2.COLOR_BGR2GRAY)
r1 = cv2.imread('man2.png',cv2.COLOR_BGR2GRAY)
l1 = cv2.cvtColor(l1,cv2.COLOR_BGR2GRAY)
r1 = cv2.cvtColor(r1,cv2.COLOR_BGR2GRAY)
# print(ssim(l2,r2))
cv2.imshow('', l1)

cv2.waitKey(0)
print(ssim(l1,r1))