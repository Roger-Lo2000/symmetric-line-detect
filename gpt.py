import cv2
import numpy as np

# 讀入影像
img = cv2.imread('man_rot90.png')

# 轉換為灰度圖像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel運算子計算
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

# 計算梯度大小和方向角度
grad_mag = cv2.magnitude(sobelx, sobely)
grad_dir = cv2.phase(sobelx, sobely, angleInDegrees=True)

# 計算每個像素與其對稱像素之間的距離，並將其加總
h, w = gray.shape
symmetric_axis_distances = []
for x in range(w // 2):
    distance_sum = 0
    for y in range(h):
        distance_sum += abs(grad_mag[y, x] - grad_mag[y, w - x - 1]) + abs(grad_dir[y, x] - grad_dir[y, w - x - 1])
    symmetric_axis_distances.append(distance_sum)

# 找到距離加總最小的位置，即為對稱軸的位置
symmetric_axis = np.argmin(symmetric_axis_distances)

# 在影像上顯示對稱軸
cv2.line(img, (symmetric_axis, 0), (symmetric_axis, h), (0, 255, 0), 2)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()