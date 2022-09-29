import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# q1


# In[2]:


### to do ---- have to write the architecture and epochs

data= pd.read_csv("smoke_detection_iot.csv")
data


# In[3]:


dataCleaned = data.drop(['UTC','Unnamed: 0'],axis=1)
dataCleaned


# In[ ]:


dataCleaned.to_numpy().shape


# In[ ]:


X = dataCleaned.drop(['Fire Alarm'],axis=1)
Y = dataCleaned['Fire Alarm']


# In[ ]:


X


# In[ ]:


X_numpy = X.to_numpy()
Y_numpy = Y.to_numpy()


# In[ ]:


X_numpy.shape


# In[ ]:


np.unique(Y_numpy)


# In[ ]:


## we are going to to create ANN which has 13 input fields and 5 hidden layers and 2 output nodes


# In[66]:


def sigmoid(x):
    x=x.astype('float')
    return 1/(1+np.exp(-x))


# In[67]:


def sigmoidDerivative(x):
    return sigmoid(x)*(1-sigmoid(x))


# In[77]:


def relu(x):
    return np.maximum(0,x)


# In[78]:


def reluDerivate(x):
    return np.greater(x,0).astype('int')


# In[68]:


def softmax(A):
    A = A.astype('float')
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


# In[69]:


def BT19ECE066_dataset_div_shuffle(csvFile,labels,ratio_training,ratio_testing):
    dataset = csvFile
    random_indexes = np.random.permutation(len(dataset))
    dataset_shuffled = dataset.iloc[random_indexes]
    Y = dataset_shuffled[labels]
    X = dataset_shuffled.drop(labels, axis=1)
    length_dataset = len(dataset)
    training_length = int(np.floor(ratio_training*length_dataset))
    testing_length = int(np.floor(ratio_testing*length_dataset))
    X_train = X[0:training_length]
    Y_train = Y[0:training_length]
    X_test = X[length_dataset-testing_length:length_dataset]
    Y_test = Y[length_dataset-testing_length:length_dataset]
    
    return (X_train,Y_train,X_test,Y_test)


# In[70]:


(X_train,Y_train,X_test,Y_test)=BT19ECE066_dataset_div_shuffle(dataCleaned,['Fire Alarm'],0.8,0.2)


# In[71]:


X_train_numpy = X_train.to_numpy()
Y_train_numpy = Y_train.to_numpy()
X_test_numpy = X_test.to_numpy()
Y_test_numpy = Y_test.to_numpy()


# In[72]:


learning_rate = 0.1


# In[82]:


W1 = np.random.rand(5,13)
b1 = np.random.randn(5,1)
W2 = np.random.rand(1,5)
b2 = np.random.randn(1,1)
epochs=10000
losses =[]
zh= 0
ah = 0
    
zo =0
ao = 0
for i in range(epochs):
    
    
    zh= np.dot(W1,X_train_numpy.T) + b1
    ah = relu(zh)
    
    zo = np.dot(W2,ah) + b2
    ao = sigmoid(zo)
#     print(ao.shape)
    
    # now doing the back prop --
    dcost_dao = ao - Y_train_numpy.T # of size (n0,m) and ah of (nh,m)
    dcost_dzo = dcost_dao
    
    dcost_dW2 = np.dot(dcost_dzo,ah.T)
    dcost_db2 = dcost_dzo
    ## w2 - (n0.nh) 
    dcost_dah = np.dot(W2.T,dcost_dzo ) # of shape ah,m
    dcost_dzh = dcost_dah*reluDerivate(zh)
    
    dcost_dW1 = np.dot(dcost_dzh,X_train_numpy)
    dcost_db1 = dcost_dzh
    
    
    ## updating with the learning parameters -
    
    W1 = W1 - (learning_rate*dcost_dW1)
    b1 = b1 - (learning_rate*dcost_db1.sum(axis=0))
    
    W2 = W2 - (learning_rate*dcost_dW2)
    b2 = b2 - (learning_rate*dcost_db2.sum(axis=0))
    
    ## finding the loss ---
    
    if i%200 ==0:
        loss = np.sum(-Y_train_numpy*np.log(ao.T))
        print("loss at iteration "+str(i)+" ----> "+str(loss))
        losses.append(loss)
        
plt.plot(losses)
    
    


# In[83]:


def calculateAccuracy(Y_true,Y_pred):
    count = 0
    for i in range(len(Y_true)):
        if(Y_true[i] == Y_pred[i]):
            count = count +1
    return count/len(Y_true)


# In[84]:


calculateAccuracy(Y_train_numpy, (np.where((ao>0.5),1,0)).flatten())


# In[85]:


np.unique((np.where((ao>0.5),1,0)).flatten())


# In[86]:


def calcuSpeciSensi(Y_true,Y_pred):
    TN = 0
    TP =0
    FP = 0
    FN = 0
    for i in range(len(Y_true)):
        if(Y_true[i] == 1):
            if (Y_pred[i]==1):
                TP = TP + 1
            else:
                FN = FN +1
        else:
            if(Y_pred[i]==1):
                FP = FP +1
            else:
                TN = TN +1
    speci = (TN)/(TN+FP)
    sensi = (TP)/(TP+FN)
    accuracy = (TN+TP)/(TN+TP+FN+FP)
    print("sensitivity is "+str(sensi))
    print("specificty is "+str(speci))
    print("accuracy is "+str(accuracy))
    


# In[87]:


calcuSpeciSensi(Y_train_numpy, (np.where((ao>0.5),1,0)).flatten())

