# In[38]:


import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


# In[2]:


data=loadmat('./Thyroid_feature_space.mat')


# In[3]:


data


# In[4]:


X=data['Thyroid_combined_feature_space']


# In[5]:


X


# In[6]:


X.shape


# In[7]:


datay=loadmat('./Thyroid_labels.mat')


# In[8]:


datay


# In[9]:


y=datay['Thyroid_target']


# In[10]:


y


# In[11]:


y.shape


# In[12]:


dic={}
count=0
for i in range(y.shape[0]):
    if dic.__contains__(y[i][0])==False:
        dic[y[i][0]]=count;
        count=count+1


# In[13]:


#Label is marked with unique id
# label 1 - 1(id)
#label 2 - 2(id)
#label 3 - 0(id)
dic


# In[14]:


for i in range(y.shape[0]):
    y[i][0]=dic[y[i][0]]


# In[15]:


y


# In[16]:


#Training Set=85% and Testing Set=15%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)


# In[17]:


X_train.shape


# In[18]:


y_train=y_train.flatten()
y_test=y_test.flatten()
print(y_train.shape,y_test.shape)


# # Start

# In[49]:


def SVC_with_kernel(C,kernel,X_train,y_train,X_test,y_test):
    model=SVC(C=C,kernel=kernel)  
    model.fit(X_train,y_train)
    print("Model Score : "+str(model.score(X_test,y_test)))
    print()
    return model


# In[50]:


def _confusion_matrix(y_pred,y_test):
    column=['3 (pred)','1 (pred)','2 (pred)']
    index=['3 (Actual)','1 (Actual)','2 (Actual)']
    cm=confusion_matrix(y_test,y_pred)
    table=pd.DataFrame(cm,columns=column,index=index)
    print(table)
    print()
    return cm


# In[51]:


def plot_ROC(tpr,fpr,label):
    print("For label : "+str(label))

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# In[52]:


def calculate_specificity_and_sensitivity(cm,label):
    TP=0
    TN=0
    FP=0
    FN=0
    if label==3:
        TP=cm[0][0]
        FN=cm[0][1]+cm[0][2]
        FP=cm[1][0]+cm[2][0]
        TN=cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2]
    if label==1:
        TP=cm[1][1]
        FN=cm[1][0]+cm[1][2]
        FP=cm[0][1]+cm[2][1]
        TN=cm[0][0]+cm[0][2]+cm[2][0]+cm[2][2]
    else:
        TP=cm[2][2]
        FN=cm[2][0]+cm[2][1]
        FP=cm[0][2]+cm[1][2]
        TN=cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1]
        
    sensitivity=TP/(TP+FP)
    specificity =TN/(TN+FN)
    print("For label "+str(label)+":\nspecificity: "+str(specificity)+"\nsensitivity: "+str(sensitivity))
    tpr=TP/(TP+FN)
    fpr=FP/(FP+TN)
    return tpr,fpr


# In[53]:


#For 3 kernels 
#1 rbf
#2 linear
#3 poly
kernels=['rbf','sigmoid','poly']
Cs=[130,130,130]


# In[55]:


for i in range(3):
    kernel=kernels[i]
    C=Cs[i]
    print("SVM with "+kernel+" kernel")
    print()
    model=SVC_with_kernel(C,kernel,X_train,y_train,X_test,y_test)
    y_pred=model.predict(X_test)
    cm=_confusion_matrix(y_pred,y_test)
    print("Confusion Matrix:")
    print(cm)
    print()
    tpr,fpr=calculate_specificity_and_sensitivity(cm,1)
    
    print(tpr,fpr)
    plot_ROC(tpr,fpr,1)
    print()
    tpr,fpr=calculate_specificity_and_sensitivity(cm,2)
    print()
    plot_ROC(tpr,fpr,2)
    print()
    tpr,fpr=calculate_specificity_and_sensitivity(cm,3)
    print()
    plot_ROC(tpr,fpr,3)
    print()

