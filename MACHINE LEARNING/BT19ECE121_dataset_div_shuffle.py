import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt


def BT19ECE121_DATASET_DIV_SHUFFLE(file_path,ratio=0.5):
    file_name, file_extension = os.path.splitext(file_path)
    if file_extension == '.csv':
        data = pd.read_csv(file_path)
    elif file_extension == '.xls':
        data = pd.read_excel(file_path)
    else:
        if file_extension == '.mat':
           data = loadmat(file_path)
           print(data.keys())
           states = data['accidents'][0][0][1]
           hdrs = data['accidents'][0][0][3]
           values = data['accidents'][0][0][2]
           data = {}
           # print(states)
           for head, val in zip(states,values):
               # print(head[0])
               # print(val)
               data[head[0][0]] = val
           # print(hdrs[0][0])
           data = pd.DataFrame.from_dict(data, orient="index", columns=[i[0] for i in hdrs[0]])
           # print(data)
            
        else:
            print('Invalid file extension')
            return
    #shuffling data
    #print(data)
    data.iloc[np.random.permutation(len(data))]
    
    
    train_data = data.iloc[0:int(len(data)*ratio),:]
    test_data = data.iloc[int(len(data)*ratio):,:]
    
    return train_data, test_data 

#BT19ECE121_DATASET_DIV_SHUFFLE(r"C:\Users\Asus\Desktop\IVLABS\MACHINE LEARNING\Matlab_accidents.mat",0.5)