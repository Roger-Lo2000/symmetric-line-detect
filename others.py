import cv2
import matplotlib.pyplot as plt
import math 
import numpy as np
class symmetry():
    def __init__(self):
        self.angle = np.zeros([512,512])
        self.r =np.zeros([512,512])
        self.bins_num = 360
        self.score = []
        self.final_angle=[]
        self.vote = []
        self.BGRimg = cv2.imread('man_rot30.png')
        self.BGRimg = cv2.resize(self.BGRimg,[512,512])
        self.img = cv2.cvtColor(self.BGRimg,cv2.COLOR_BGR2GRAY)
        
        self.dx = cv2.Sobel(self.img, cv2.CV_32F,1,0)
        self.dy = cv2.Sobel(self.img, cv2.CV_32F,0,1)
        # self.dx = cv2.convertScaleAbs(self.dx)
        # self.dy = cv2.convertScaleAbs(self.dy)
        for i in range(512):
            for j in range(512):
                self.angle[i][j] = round((math.atan2(self.dy[i][j],self.dx[i][j]))/math.pi*180,2)
                if self.angle[i][j]<0:
                    self.angle[i][j]+=180
                # if self.angle[i][j] == 180:
                #     a = cv2.circle(self.BGRimg,(i,j),1,(0,255,0),-1)
        # print(self.angle)
        
        n,bins,patches = plt.hist(self.angle.ravel(),bins = self.bins_num)
        bins = np.round(bins,2)
        for i in range(self.bins_num):
            temp= 0
            for j in range(int(self.bins_num/2)):
                if i-j<0:
                    temp += n[i+j]*n[i-j+self.bins_num]
                elif(i+j>=self.bins_num):
                    temp += n[i+j-self.bins_num]*n[i-j]
                else:
                    temp += n[i+j]*n[i-j]
            self.score.append(temp)
            
        self.score_sorted = sorted(self.score)
        print(self.score_sorted[-2:])
        for i in self.score_sorted[-2:]:
            # print(self.score.index(i))
            # print(bins[self.score.index(i)])
            self.final_angle.append(bins[self.score.index(i)])
        print(self.final_angle)
        
        ret,thresh = cv2.threshold(self.img,127,255,0)
        M = cv2.moments(thresh)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.line(self.BGRimg, (cX, cY), (int(cX*math.sin(90)),int(cY*math.cos(90))), (255, 255, 255), 1)
        plt.show()
        cv2.imshow('',self.BGRimg)
        cv2.waitKey()
        
Symmetry = symmetry()