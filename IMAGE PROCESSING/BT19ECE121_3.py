# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 01:04:07 2022

@author: Asus

"""

import numpy as np
from PIL import Image

def gamma_transform(input_image='cameraman.jpeg',gamma=0):
    input_image = Image.open(input_image)
    input_image = np.float64(input_image)
    dimension = input_image.ndim
    m = input_image.shape[0]
    n = input_image.shape[1]
    for i in range(0,m):
        for j in range(0,n):
            if dimension==3:
                r=input_image.shape[2]
                for k in range(0,r):
                    input_image[i][j][k]/=255
                    input_image[i][j][k]**=gamma
                    input_image[i][j][k]*=255
            else:
                input_image[i][j]/=255
                input_image[i][j]**=gamma
                input_image*=255
    #print(input_image.shape)
    return input_image

def contrast_streching(input_image='cameraman.jpeg',min_val_r=0,max_val_r=0,min_val_s=0,max_val_s=0):
    input_image=Image.open(input_image)
    input_image=np.float64(input_image)
    dimension=input_image.ndim
    m=input_image.shape[0]
    n=input_image.shape[1]
    slope1=min_val_s/min_val_r
    slope2=(max_val_s-min_val_s)/(max_val_r-min_val_r)
    slope3=(255-max_val_s)/(255-max_val_r)
    y_intercept_1=min_val_s-slope2*min_val_r
    y_intercept_2=max_val_s-slope3*max_val_r
    for i in range(0,m):
        for j in range(0,n):
            if dimension==3:
                r=input_image.shape[2]
                for k in range(0,r):
                    if input_image[i][j][k]>=0 and input_image[i][j][k]<=min_val_r:
                        input_image[i][j][k]*=slope1
                    if input_image[i][j][k]>min_val_r and input_image[i][j][k]<=max_val_r:
                        input_image[i][j][k]=input_image[i][j][k]*slope2+y_intercept_1
                    else:
                        input_image[i][j][k]=input_image[i][j][k]*slope3+y_intercept_2

            else:
                if input_image[i][j] >= 0 and input_image[i][j] <= min_val_r:
                    input_image[i][j]*= slope1
                if input_image[i][j] > min_val_r and input_image[i][j]<= max_val_r:
                    input_image[i][j] = input_image[i][j]* slope2 + y_intercept_1
                else:
                    input_image[i][j] = input_image[i][j]* slope3 + y_intercept_2
    #print(input_image.shape)         
    return input_image
    

#print(contrast_streching('cameraman.jpeg',40,100,70,200))
#print(gamma_transform('cameraman.jpeg',gamma=0.8))
