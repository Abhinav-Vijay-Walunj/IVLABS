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
            # con_list = [[element for element in headers] for headers in data]
            print(data.keys())
            states = data['accidents'][0][0][1]
            header = data['accidents'][0][0][3]
            values = data['accidents'][0][0][2]
            data = {}
            # print(states)
            for head, val in zip(states,values):
                # print(head[0])
                # print(val)
                data[head[0][0]] = val
            # print(header[0][0])
            data = pd.DataFrame.from_dict(data, orient="index", columns=[i[0] for i in header[0]])
            # print(data)
            # data:2
            # data = pd.DataFrame.from_dict(list(con_list))
        else:
            print('Invalid file extension')
            return
    #shuffling data
    #print(data)
    data.iloc[np.random.permutation(len(data))]
    
    #spliting data into train and test set
    train_data = data.iloc[0:int(len(data)*ratio),:]
    test_data = data.iloc[int(len(data)*ratio):,:]
    
    return train_data, test_data 
        
def visulization(X, Y, pred = None, scatter = True):
    plt.figure()
    if(scatter):
        plt.plot(X, Y, 'ro')
    else:
        plt.plot(X, Y, 'ro')
        plt.plot(X, pred)
    # plt.show()

def pseudo_inverse(train_x,y):
    return np.linalg.pinv(train_x.T @ train_x) @ train_x.T @ y




def train_linear_reg(x, train_x, train_y, pseudo_weight, learning_rate = 0.0001, iterations = 100):
    plt.figure()
    plt.plot(x, train_x@pseudo_weight)
    
    weight = np.random.rand(train_x.shape[1],1)
    loss_over_time = []
    max_x = np.max(train_x[:,1])
    normalize_train_x = train_x
    normalize_train_x[:,1]= normalize_train_x[:,1]/max_x
    max_y = np.max(train_y)
    normalize_train_y = train_y/max_y
    plt.plot(max_x*normalize_train_x[:,1], max_y*normalize_train_y, 'ro')
    # print(normalize_train_x, normalize_train_y)
    for i in range(iterations):
        pred = normalize_train_x @ weight
        diff = pred - normalize_train_y.reshape(-1,1)
        loss = np.sum(diff**2)/len(normalize_train_y)
        dl_da = 2/len(normalize_train_y)
        da_ddiff  = diff
        ddiff_dy = 1
        ddiff_dw= normalize_train_x.T
        dldw = dl_da * ddiff_dy*(ddiff_dw @ da_ddiff)
        # plt.figure()
        # print("normalize_train_x", normalize_train_x.shape)
        # print("normalize_train_y", normalize_train_y.shape)
        # print("pred:", pred.shape)
        # print("diff:", diff.shape)
        # print("loss:", loss.shape)
        # print("dldw:", dldw.shape)
        # print("weight:", weight.shape)
        # print("dldw:", dldw)
        # print()
        # print(pred.shape, normalize_train_x.shape, diff.shape, loss.shape, dl_da.shape, da_ddiff.shape, ddiff_dy.shape, ddiff_dw.shape, dldw.shape)
        # plt.plot(pred, normalize_train_x[:,1])
        # plt.show()
        # plt.pause(0.001)
        weight = weight - learning_rate*dldw
        # loss = np.sum(pred - normalize_train_y)
        loss_over_time.append(np.mean(loss))
    plt.plot(max_x*normalize_train_x[:,1], max_y*(normalize_train_x @ weight))
    plt.legend(["Pseudo Inverse","Data", "Linear Regression"])
    plt.figure()
    plt.plot(loss_over_time)
    plt.legend(["Cost function"])
    # plt.show()
    return weight

#here below implemented gradient descent
def mean_squared_error(y_true, y_predicted):
     
    # Calculating the loss or cost
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost
 
# Gradient Descent Function
# Here iterations, learning_rate, stopping_threshold
# are hyperparameters that can be tuned
def gradient_descent(x, y, iterations = 1000, learning_rate = 0.0001,
                     stopping_threshold = 1e-6):
     
    # Initializing weight, bias, learning rate and iterations
    current_weight = 0.1
    current_bias = 0.01
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))
     
    costs = []
    weights = []
    previous_cost = None
     
    # Estimation of optimal parameters
    for i in range(iterations):
         
        # Making predictions
        y_predicted = (current_weight * x) + current_bias
         
        # Calculationg the current cost
        current_cost = mean_squared_error(y, y_predicted)
 
        # If the change in cost is less than or equal to
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break
         
        previous_cost = current_cost
 
        costs.append(current_cost)
        weights.append(current_weight)
         
        # Calculating the gradients
        weight_derivative = -(2/n) * sum(x * (y-y_predicted))
        bias_derivative = -(2/n) * sum(y-y_predicted)
         
        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
                 
        # Printing the parameters for each 1000th iteration
        print(f"Iteration {i+1}: Cost {current_cost}, Weight \
        {current_weight}, Bias {current_bias}")
     
     
    # Visualizing the weights and cost at for all iterations
    plt.figure(figsize = (8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()
     
    return current_weight, current_bias
 
    
 
    

def BT19ECE121_linreg(filepath):
    train,test = BT19ECE121_DATASET_DIV_SHUFFLE(filepath,0.5)
    x,y = train.iloc[:,6].to_numpy(),train.iloc[:,5].to_numpy()
    # print(x,y)
    # visulization(x,y)
    train_x  = np.hstack([np.ones((len(x),1)),np.array(x).reshape(-1,1)])
    # print(train_x)
    weights = pseudo_inverse(train_x,y)
    # print(weights)
    # visulization(x,y,pred = train_x @ weights, scatter = False)
    train_weights = train_linear_reg(x, train_x, y, weights,learning_rate = 0.01, iterations = 1000)
    # visulization(x,y,pred = train_x @ train_weights, scatter = False)
    
    #here below implement gradient descent
    estimated_weight, eatimated_bias = gradient_descent(x, y, iterations=2000)
    print(f"Estimated Weight: {estimated_weight}\nEstimated Bias: {eatimated_bias}")
 
    # Making predictions using estimated parameters
    Y_pred = estimated_weight*x + eatimated_bias
 
    # Plotting the regression line
    plt.figure(figsize = (8,6))
    plt.scatter(x, y, marker='o', color='red')
    plt.plot([min(x), max(x)], [min(Y_pred), max(Y_pred)], color='blue',markerfacecolor='red',
             markersize=10,linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    
#BT19ECE121_linreg(r"C:\Users\Asus\Desktop\IVLABS\MACHINE LEARNING\Matlab_accidents.mat")