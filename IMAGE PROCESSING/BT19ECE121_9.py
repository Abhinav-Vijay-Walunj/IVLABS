import numpy as np
from PIL import Image

def summation0deg(matrix,idx,x):
    summation=0
    if x==1:
        summation=matrix[idx][0]+matrix[idx][1]+matrix[idx][2]
    else:
        summation=matrix[0][idx]+matrix[1][idx]+matrix[2][idx]
    return summation

def summation45deg(matrix,idx,idy,x):
    if x==1:
        summation=0
        while idx>=0 and idy<3:
            summation+=matrix[idx][idy]
            idx=idx-1
            idy=idy+1
        return summation
    else:
        summation=0
        while idx>=0 and idy>=0:
            summation+=matrix[idx][idy]
            idx=idx-1
            idy=idy-1
        return summation


def Radon_Transform(input_matrix):    
    rows, cols = (5, 5)
    img = np.zeros((5,5))

    img[1][0]= summation0deg(input_matrix,0,1)
    img[2][0] = summation0deg(input_matrix,1,1)
    img[3][0] = summation0deg(input_matrix,2,1)
    
    for i in range(5):
        idx=min(2,i)
        idy=0
        if i>=3:
            idy=i-2
        img[i][1]=summation45deg(input_matrix,idx,idy,1)
        
    img[1][2] = summation0deg(input_matrix,0,-1)
    img[2][2] = summation0deg(input_matrix,1,-1)
    img[3][2] = summation0deg(input_matrix,2,-1)

    for i in range(4,-1,-1):
        idx=min(i,2)
        idy=0
        if i>=2:
            idy=4-i
        else:
            idy=2
#         print(idx,idy)
        img[4-i][3]=summation45deg(input_matrix,idx,idy,-1)

    img[1][4]=img[1][0]
    img[2][4] = img[2][0]
    img[3][4] = img[3][0]
    return img