# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 21:57:07 2022

@author: Asus
"""

#import matplotlib.pyplot as plt
#import numpy as np
#from PIL import Image

def calculate_metrics(ndarray):
    size=ndarray.ndim
    if size==3:
        m, n,r = ndarray.shape
        
        sum = np.array([0]*3)
        square_difference_sum = np.array([0]*3)
        hst_3d=np.zeros(shape=(256,3))
        mean3d = sum / (m*n)
        for i in range(0, m):
            for j in range(0, n):
                for k in range(0,r):
                    square_difference_sum[k] += (ndarray[i][j][k] - mean3d[k]) ** 2

        for i in range(0,m):
            for j in range(0,n):
                for k in range(0,r):
                    sum[k]+=ndarray[i][j][k]
                    hst_3d[ndarray[i][j][k]][k]+=1
        
        
        for i in range(0,3):
            square_difference_sum[i]=(square_difference_sum[i]/(m*n))**0.5
        
        mean=np.array([mean3d])
        standard_dev=np.array([square_difference_sum])
        hst=np.array([hst_3d])
        
        return mean, standard_dev, hst
        

    else:
        
        m,n=ndarray.shape
        square_difference_sum=0
        sum=0
        hst_2d = [0] * 256
        for i in range(0,m):
            for j in range(0,n):
                sum+=ndarray[i][j]
                hst_2d[ndarray[i][j]]+=1

        for i in range(0,256):
            hst_2d[i]/=m*n

        mean2d=sum/(m*n)
        
        for i in range(0,m):
            for j in range(0,n):
                square_difference_sum+=(ndarray[i][j]-mean2d)**2

        standard_dev_2d=(square_difference_sum/(m*n))**0.5
        mean=np.array([[mean2d]])
        standard_dev=np.array([[standard_dev_2d]])
        hst=np.array([hst_2d])
        
        return mean,standard_dev,hst
    
#img=Image.open('cameraman.jpeg')
#ndarray=np.array(img)
#x=calculate_metrics(ndarray)
#print(x)