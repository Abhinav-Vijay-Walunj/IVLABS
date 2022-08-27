# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 22:57:03 2022

@author: Asus
"""



import numpy as np
from PIL import Image










def rotatekernel_mask(given_filter):
    m,n=given_filter.shape
    out=given_filter
    for i in range(m):
        for j in range(n):
            out[i, n-1-j] = given_filter[m-1-i, j]
    return out













def convolution2d(given_image,given_filter=np.array([]),padding=np.array([]),stride=np.array([])):
    given_filter=rotatekernel_mask(given_filter)
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
    convoluted_final_image=[]
    if dimen==2:
        convoluted_final_image = []
        for i in range(0, m - n1 + 1, stride_in_y):
            op = []
            for j in range(0, n - n2 + 1, stride_in_x):
                value = 0
                for p in range(n1):
                    for q in range(n2):
                        value += given_filter[p][q] * img[i + p][j + q]
                op.append(value)
            convoluted_final_image.append(op)
        convoluted_final_image=np.array(convoluted_final_image)
        



    else:
        o=given_image.shape[2]
        convoluted_final_image=[]
        for i in range(0,m-n1+1,stride_in_y):
            op=[]
            for j in range(0,n-n2+1,stride_in_x):
                value = [0,0,0]
                for k in range(o):
                    for p in range(n1):
                        for q in range(n2):
                            value[k]+=given_filter[p][q]*img[i+p][j+q][k]
                op.append(value)    
            convoluted_final_image.append(op)
    convoluted_final_image=np.array(convoluted_final_image)
    convoluted_final_image=convoluted_final_image.astype('uint8')
    return convoluted_final_image

#given_filter=np.array([[1,0,-1],[0,0,0],[-1,0,1]])
#padding=np.array([[1],[2]])
#stride=np.array([[2],[1]])
#given_image = Image.open('cameraman.jpeg')
#given_image=np.array(given_image)

#convoluted_final_image=convolution2d(given_image,given_filter,padding,stride)
#print(convoluted_final_image)