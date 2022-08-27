# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 22:57:30 2022

@author: Asus
"""

import numpy as np
from PIL import Image











def correlation2d(given_image,given_filter=np.array([]),padding=np.array([]),stride=np.array([])):
    
    dimen=given_image.ndim

    # padding
    padding_in_x=padding[0][0]
    padding_in_y=padding[1][0]
    y=padding_in_x
    x=padding_in_y
    m=given_image.shape[0]
    n=given_image.shape[1]
    
    
    
    
    if dimen == 3:
        o=given_image.shape[2]
        img=np.zeros((m+2*padding_in_y,n+2*padding_in_x,o))
        for i in range(m):
            for j in range(n):
                for k in range(o):
                    img[i+x][j+y][k]=given_image[i][j][k]
    else:
        img = given_image.zeros((m + 2 * padding_in_y, n + 2 * padding_in_x))
        for i in range(m):
            for j in range(n):
                img[i+x][j+y]=given_image[i][j]



    #striding
    stride_in_x=stride[0][0]
    stride_in_y=stride[1][0]
    n1,n2=given_filter.shape
    m=img.shape[0]
    n=img.shape[1]
    correlatade_final_image=[]
    
    
    
    if dimen==2:
        correlatade_final_image = []
        for i in range(0, m - n1 + 1, stride_in_y):
            op = []
            for j in range(0, n - n2 + 1, stride_in_x):
                value = 0
                for p in range(n1):
                    for q in range(n2):
                        value += given_filter[p][q] * img[i + p][j + q]
                op.append(value)
            correlatade_final_image.append(op)
        correlatade_final_image=np.array(correlatade_final_image)
        



    else:
        o=given_image.shape[2]
        correlatade_final_image=[]
        for i in range(0,m-n1+1,stride_in_y):
            op=[]
            for j in range(0,n-n2+1,stride_in_x):
                value = [0,0,0]
                for k in range(o):
                    for p in range(n1):
                        for q in range(n2):
                            value[k]+=given_filter[p][q]*img[i+p][j+q][k]
                op.append(value)    
            correlatade_final_image.append(op)
            
            
    correlatade_final_image=np.array(correlatade_final_image)
    correlatade_final_image=correlatade_final_image.astype('uint8')
    return correlatade_final_image


