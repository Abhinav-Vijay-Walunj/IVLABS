# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 20:58:01 2022

@author: Asus
"""


import numpy as np
from PIL import Image
from math import inf





def get_range(input_image):
    dimen=input_image.ndim
    if dimen==2:
        maxx0 = -float('inf')
        minn0=float('inf')
        m,n=input_image.shape
        for i in range(0,m):
            for j in range(0,n):
                maxx0=max(maxx0,input_image[i][j])
                minn0=min(minn0,input_image[i][j])
        return np.array([[minn0,maxx0]])
    else:
        maxx0 = -float('inf')
        minn0 = float('inf')
        maxx1 = -float('inf')
        minn1 = float('inf')
        maxx2 = -float('inf')
        minn2 = float('inf')
        m, n,r = input_image.shape
        for i in range(0, m):
            for j in range(0, n):
                for k in range(0,r):
                    if k==0:
                        maxx0 = max(maxx0, input_image[i][j][k])
                        minn0 = min(minn0, input_image[i][j][k])
                    if k==1:
                        maxx1 = max(maxx1, input_image[i][j][k])
                        minn1 = min(minn1, input_image[i][j][k])
                    if k==2:
                        maxx2 = max(maxx2, input_image[i][j][k])
                        minn2 = min(minn2, input_image[i][j][k])
        return np.array([[minn0,maxx0],[minn1,maxx1],[minn2,maxx2]])





def get_variance(input_image,mean):
    dimen=input_image.ndim
    if dimen==2:
        sum=np.array([0])
        m,n=input_image.shape
        for i in range(0,m):
            for j in range(0,n):
                sum[0]+=(input_image[i][j]-mean[0])**2
        sum[0]/=m*n-1
        return np.array([sum])
    else:
        sum = np.array([0]*3)
        m, n,r = input_image.shape
        for i in range(0, m):
            for j in range(0, n):
                for k in range(0,r):
                    sum[k] += (input_image[i][j][k] - mean[k]) ** 2
        sum[0]/=m*n-1
        sum[1]/=m*n-1
        sum[2]/=m*n-1
        return np.array([sum])  
    
    
    
    
    
    
    
def mean_and_standard(ndarray):

    dimension=ndarray.ndim
    
    
    
    
    
    if dimension==3:
        # 3d array -> colour image
        m, n,r = ndarray.shape
        squared_different_sum = np.array([0]*3)
        sum = np.array([0]*3)
        # print(histogram_3d.shape)
        for i in range(0,m):
            for j in range(0,n):
                for k in range(0,r):
                    sum[k]+=ndarray[i][j][k]




        mean3d = sum / (m*n)
        for i in range(0, m):
            for j in range(0, n):
                for k in range(0,r):
                    squared_different_sum[k] += (ndarray[i][j][k] - mean3d[k]) ** 2
        for i in range(0,3):
            squared_different_sum[i]=(squared_different_sum[i]/(m*n))**0.5
        mean=np.array(mean3d)
        standard_deviation=np.array(squared_different_sum)
        return mean, standard_deviation

    else:
        # 2d array -> greyscale image
        m,n=ndarray.shape
        squared_different_sum=0
        sum=0
        for i in range(0,m):
            for j in range(0,n):
                sum+=ndarray[i][j]

        mean2d=sum/(m*n)
        for i in range(0,m):
            for j in range(0,n):
                squared_different_sum+=(ndarray[i][j]-mean2d)**2

        standard_deviation_2d=(squared_different_sum/(m*n))**0.5
        mean=np.array([mean2d])
        standard_deviation=np.array([standard_deviation_2d])
        return mean,standard_deviation
        

        






def normalisation(input_image,pn=False,pc=True,ps=False):
    dimen=input_image.ndim
    normalised_output_image=np.array([])
    
    
    
    
    
    
    
    #Pixel Normalisation
    if pn==True:
        normalised_output_image = input_image / 255

    mean,std_dev=mean_and_standard(input_image)
    # print(mean)
    # print(std_dev)
    
    
    
    
    
    
    
    
    #Pixel Centering
    if pc==True:
        if dimen == 2:
            m,n=input_image.shape
            normalised_output_image=np.zeros([m,n])
            for i in range(0,m):
                for j in range(0,n):
                    normalised_output_image[i][j]=input_image[i][j]-mean[0]
        else:
            m, n,r = input_image.shape
            normalised_output_image = np.zeros([m, n,r])
            print(normalised_output_image.shape)
            for i in range(0, m):
                for j in range(0, n):
                    for k in range(0,r):
                        normalised_output_image[i][j][k] = input_image[i][j][k] - mean[k]





    
    #Pixel Standardization
    if ps==True:
        if dimen == 2:
            m, n = input_image.shape
            normalised_output_image = np.zeros([m, n])
            for i in range(0, m):
                for j in range(0, n):
                    normalised_output_image[i][j] = (input_image[i][j] - mean[0])/std_dev[0]
        else:
            m, n, r = input_image.shape
            normalised_output_image = np.zeros([m, n, r])
            print(normalised_output_image.shape)
            for i in range(0, m):
                for j in range(0, n):
                    for k in range(0, r):
                        normalised_output_image[i][j][k] = (input_image[i][j][k] - mean[k])/std_dev[k]
        
    
    range_output_image=get_range(normalised_output_image)
    variance_output_image=get_variance(normalised_output_image,mean)
    return normalised_output_image,np.array([mean]),range_output_image,variance_output_image

















#x = Image.open('cameraman.jpeg')
#img=np.array(x)
#print(img.shape)
#print(normalisation(img,pc=True))