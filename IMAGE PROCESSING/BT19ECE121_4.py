# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 22:01:30 2022

@author: Asus
"""

import numpy as np
from PIL import Image


def histogram_equalization(input_image='location of input image'):
    input_image = np.array(Image.open(input_image))
    dimen = input_image.ndim
    
    
    if dimen == 2:
        n = input_image.shape[0]
        m = input_image.shape[1]
        output_image = np.zeros((n, m))
        frequency_plot = np.zeros(266)
        for i in range(n):
            for j in range(m):
                frequency_plot[input_image[i][j]]+=1


        for i in range(266):
            frequency_plot[i]/=m*n


        for i in range(1, 266):
            frequency_plot[i] += frequency_plot[i - 1]


        for i in range(266):
            frequency_plot[i]*= 255
            frequency_plot[i]= int(frequency_plot[i])


        for i in range(n):
            for j in range(m):
                output_image[i][j]=frequency_plot[input_image[i][j]]
    
        
    else:
        n = input_image.shape[0]
        m = input_image.shape[1]
        t = input_image.shape[2]
        
        
        
        output_image = np.zeros((n, m, t))
        frequency_plot=np.zeros((3,266))
        for i in range(n):
            for j in range(m):
                for k in range(t):
                    frequency_plot[k][input_image[i][j][k]]+=1
                    
        
        
        for i in range(3):
            for j in range(266):
                frequency_plot[i][j]/=m*n
                #print(frequency_plot[i][j])

        for i in range(3):
            for j in range(1,266):
                frequency_plot[i][j]+=frequency_plot[i][j-1]
            #print(frequency_plot[i][255])

        for i in range(3):
            for j in range(266):
                frequency_plot[i][j]*=255
                frequency_plot[i][j]=int(frequency_plot[i][j])

        for i in range(n):
            for j in range(m):
                for k in range(t):
                    output_image[i][j][k]=frequency_plot[k][input_image[i][j][k]]

    return(output_image)









def gray_slicing(input_image='location of input image', thr1=0, thr2=0):
    input_image = Image.open(input_image)
    input_image=np.array(input_image)
    dimen=input_image.ndim
    # print(type(input_image))
    if dimen==2:
        
        n = input_image.shape[0]
        m = input_image.shape[1]
        
        
        Output_images_with_Background_present = np.zeros((n, m))
        Output_images_when_Background_absent = np.zeros((n, m))
        
        
        for i in range(n):
            for j in range(m):
                Output_images_with_Background_present[i][j] = input_image[i][j]
                Output_images_when_Background_absent[i][j] = 255
                
                if input_image[i][j] >= thr1 and input_image[i][j] <= thr2:
                    Output_images_with_Background_present[i][j] = 255
                    Output_images_when_Background_absent[i][j] = 0
    
        
    else:
        n=input_image.shape[0]
        m=input_image.shape[1]
        t=input_image.shape[2]
        
        Output_images_with_Background_present=np.zeros((n,m,t))
        Output_images_when_Background_absent=np.zeros((n,m,t))
        
        for i in range(n):
            for j in range(m):
                for k in range(t):
                    Output_images_with_Background_present[i][j][k]=input_image[i][j][k]
                    Output_images_when_Background_absent[i][j][k]=255
                    
                    if input_image[i][j][k]>=thr1 and input_image[i][j][k]<=thr2:
                        Output_images_with_Background_present[i][j][k]=255
                        Output_images_when_Background_absent[i][j][k] = 0

    return(Output_images_with_Background_present, Output_images_when_Background_absent)





#print(gray_slicing("cameraman.jpeg",20,150))

#print(histogram_equalization("cameraman.jpeg"))
