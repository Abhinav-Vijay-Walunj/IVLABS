import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow import keras
import sklearn
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt

data=loadmat(r"C:\Users\Asus\Desktop\IVLABS\MACHINE LEARNING\Matlab_cancer.mat")
data['x'].shape

df=pd.DataFrame(data['x'].T)

df['result'] = data['t'][0]
df.head(10)


X=df.drop(['result'],axis=1)

X



