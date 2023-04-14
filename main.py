import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
def rotate_image(image, angle): ## rotate left degrees
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderMode= cv2.BORDER_REPLICATE)
  return result

def main(input_filen):
    radian = 180 / math.pi
    response_threshold = 0
    # 讀取影像
    img_color = cv2.imread(input_file)
    img_color = cv2.resize(img_color, (450, 450), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
    img_f = img.astype(np.float32)
    hist = np.zeros((img.shape[0],img.shape[1]),dtype = int)
    # 初始化梯度方向柱狀圖
    x_axis = np.arange(0,180,1,dtype=int)
    # 計算image的x gradient跟y gradient
    k = 3
    dst_x = cv2.Sobel(img_f,cv2.CV_32F,1,0,ksize=k)
    dst_y = cv2.Sobel(img_f,cv2.CV_32F,0,1,ksize=k)
    shape = dst_x.shape
    # 根據每一點計算response跟angle，計算出angle histogram
    for i in range(shape[0]):
        for j in range(shape[1]):
            angle = math.atan2(dst_y[i][j],dst_x[i][j]) * radian
            if(angle < 0):
                angle += 180
            if(angle == 180):
                angle = 0
            hist[i][j] = int(angle)
    hist = hist.ravel()

    ## 建立correlation function 
    angle = np.zeros(180)
    for i in range(len(hist)):
        angle[hist[i]] += 1
    S = np.zeros(180)
    for i in range(len(S)): ## x
        for j in range(len(S)): ## theta
            theta1 = i + j
            theta2 = i - j
            if(theta1 >= 180):
                theta1 -= 180
            if(theta2 < 0):
                theta2 += 180
            S[i] += angle[theta1] * angle[theta2]
    ## 畫出histogram 和 correlation function
    plt.subplot(1, 2, 1)
    n, bins, pathes = plt.hist(hist, bins = 720)
    plt.title("histogram", {'fontsize':15})
    plt.subplot(1,2,2)
    plt.plot(S)
    plt.title("correlation function", {'fontsize':15})
    plt.show()

    # 得到候選對稱軸角度
    symmetric_angle_list = []
    for i in range(len(S)):
        symmetric_angle_list.append([i,S[i]])
    symmetric_angle_list = sorted(symmetric_angle_list,key=lambda l:l[1], reverse=True)

    ## 根據候選對稱軸進行鏡像反轉，測試SSIM
    max_SSIM = 0
    max_ang = symmetric_angle_list[0][0]
    for i in range(30): 
        ang = symmetric_angle_list[i][0]
        img_rot = rotate_image(img,90-ang)
        left_side = img_rot[:,0:shape[0]//2]
        right_side = img_rot[:,shape[0]//2:]
        right_side = cv2.flip(right_side, 1)
        ## flip right_side image
        ssim_val = ssim(right_side,left_side)
        print("SSIM:",ssim_val, "test angle:", ang)
        if(ssim_val > max_SSIM):
            max_SSIM = ssim_val
            max_ang = ang
    
    print("symmetric angle = ", max_ang)


    ## 得到影像重心
    image_sum = np.sum(img_f)
    image_x_sum = 0
    image_y_sum = 0

    t = 30
    for i in range(shape[0]):
        for j in range(shape[1]):
            if(img[i][j] > t):
                image_x_sum += img_f[i][j] * i
                image_y_sum += img_f[i][j] * j
    center_x = image_x_sum / image_sum
    center_y = image_y_sum / image_sum

    ## 根據影像重心以及對稱軸角度畫線
    print("center point = ", (int(center_x),int(center_y)))
    for i in range(5):
        x1 = 0
        y1 = 0
        length = shape[0] * 1.414
        x1 = int(center_x + length * math.cos(max_ang / radian))
        y1 = int(center_y - length * math.sin(max_ang / radian))
        x2 = int(center_x - length * math.cos(max_ang / radian))
        y2 = int(center_y + length * math.sin(max_ang / radian))
        cv2.line(img_color,(x1,y1),(x2,y2),(0,0,255),2)
    cv2.imshow('color',img_color)
    cv2.imwrite('8.png',img_color)
    cv2.waitKey(0)

if __name__ == '__main__':
    input_file = 'man_rot0.png'
    main(input_file)