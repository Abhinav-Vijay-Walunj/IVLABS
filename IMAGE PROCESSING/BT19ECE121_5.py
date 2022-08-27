# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 23:02:23 2022

@author: Asus
"""

import numpy as np
import math
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt



def scale_image(input_image='location of input image', scaling_fact_alpha=0):
    
    input_image = Image.open(input_image)
    input_image=np.array(input_image)
    
    n = input_image.shape[0]
    m = input_image.shape[1]
    
    
    N=scaling_fact_alpha*n
    M=scaling_fact_alpha*m
    
    
    output_image=np.zeros((N,M))



    for i in range(N):
        for j in range(M):
            output_image[i][j]=-1



    for i in range(n):
        for j in range(m):
            output_image[i*scaling_fact_alpha][j*scaling_fact_alpha]=input_image[i][j]



    for i in range(0,N,scaling_fact_alpha):
        given_input=output_image[i][0]
        for j in range(0,M):
            if output_image[i][j]==-1:
                output_image[i][j]=given_input
            else:
                given_input=output_image[i][j]

        for k in range(i+1,i+scaling_fact_alpha):
            for j in range(M):
                output_image[k][j]=output_image[i][j]

    return (output_image)



def rotate_image(input_image='location of input image', angle=45):
    img = Image.open(input_image)
    img=np.array(img)
    plt.imshow(img)
    rotation_degree = angle


    rotation_in_radian = rotation_degree * np.pi / 180.0



    height, width = img.shape
    num_channels=1


    max_len = int(math.sqrt(height*height + width*width))
    rotated_image = np.zeros((max_len, max_len, num_channels))



    height_after_rotation, width_after_rotation, _ = rotated_image.shape
    middle_row = int( (height_after_rotation+1)/2)
    middle_coloumn = int( (width_after_rotation+1)/2)


    for r in range(height_after_rotation):
        for c in range(width_after_rotation):
            y = (r-middle_coloumn)*math.cos(rotation_in_radian) + (c-middle_row)*math.sin(rotation_in_radian)
            x = -(r-middle_coloumn)*math.sin(rotation_in_radian) + (c-middle_row)*math.cos(rotation_in_radian)
            y += middle_coloumn
            x += middle_row
            x = round(x)
            y = round(y)
            if (x >= 0 and y >= 0 and x < width and y < height):
                rotated_image[r][c] = img[y][x]

    output_image=np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            output_image[i][j]=rotated_image[i][j][0]

    return output_image

#print(scale_image("cameraman.png",2))
#print(rotate_image("cameraman.png",45))