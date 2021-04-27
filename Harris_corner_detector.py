import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
%matplotlib inline


def corner_rsd(input_img_rsd, b_size = 9, k_size=3, alpha = 0.04):
    # inputs: 
    # input_img_rsd: input grayscale image
    # b_size: block_size for Gaussian filter
    # k_size: k_size for sobel i.e. sobel window size
    # alpha: constant in R 
    # for a grayscale image finds harris corner
    if input_img_rsd.ndim == 3:
        input_img_rsd = cv2.cvtColor(input_img_rsd, cv2.COLOR_BGR2GRAY)
    h, w = input_img_rsd.shape[0],input_img_rsd.shape[1]
            
    # derivatives using sobel operator
    der_x = cv2.Sobel(input_img_rsd,cv2.CV_64F,1,0,ksize=k_size)
    der_y = cv2.Sobel(input_img_rsd,cv2.CV_64F,0,1,ksize=k_size)

    # 2nd moment matrix generation
    Ixx = der_x*der_x
    Ixy = der_x*der_y
    Iyy = der_y*der_y
            
    # gaussian kernel size is given by ip b_size

    # M = summation of W(x,y)[[Ix^2 ,IxIy],[IxIy,Iy^2]]
    Ixx = cv2.GaussianBlur(Ixx,(b_size,b_size),0)
    Ixy = cv2.GaussianBlur(Ixy,(b_size,b_size),0)
    Iyy = cv2.GaussianBlur(Iyy,(b_size,b_size),0)
    # r matrix
    r_mat = np.zeros([h,w],dtype=float)
    for i in range(0,h):
        for j in range(0,w):
            M = [[Ixx[i,j],Ixy[i,j]],[Ixy[i,j],Iyy[i,j]]]

            r_mat[i][j] = np.linalg.det(M) - alpha*(np.trace(M)**2)
    # threshold = 1% Rmax
    threshold= 0.01*np.max(r_mat) 
    new_img = input_img_rsd.copy()
    new_img=cv2.cvtColor(new_img, cv2.COLOR_GRAY2RGB)
    for i in range(0,len(r_mat)):
        for j in range(0,len(r_mat[0])):
            if r_mat[i,j] >= threshold:
                cv2.circle(new_img,(j,i),1,(255,255,0),-1)
    corners_rsd = []
    for i in range(h):
        for j in range(w):
            if r_mat[i][j] >= threshold:
                corners_rsd.append([i,j]) 
            # returns list with corners coordinates
    return corners_rsd, new_img
