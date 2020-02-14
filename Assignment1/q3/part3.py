
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import math
import sys


# In[2]:




# In[3]:


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    data = data -mean
    data = data/std
    return data


# In[4]:


def inputData():
    dfX1 = pd.read_csv(sys.argv[1],usecols=[0],names=["X1"],header=None)
    dfX2 = pd.read_csv(sys.argv[1],usecols=[1],names=["X2"],header=None)
    dfY = pd.read_csv(sys.argv[2],usecols=[0],names=["Y"],header=None)
    #creating the intercept term
    X_0 = np.ones((len(dfX1),1))
    #normalizing the data
    X_1 = normalize(dfX1["X1"].to_numpy()).reshape(-1,1)
    X_2 = normalize(dfX2["X2"].to_numpy()).reshape(-1,1)
#     print(X_1.shape)
    Y = dfY.to_numpy().reshape(-1,1)
    #joining the training example as one numpy Narray
    X0X1 = np.append(X_0,X_1, axis=1)
    X0X1X2 = np.append(X0X1,X_2,axis = 1)
    X0X1X2Y = np.append(X0X1X2,Y, axis=1)
    np.random.shuffle(X0X1X2Y)  #shuffling data to make it random for better distribution
#     print(X0X1X2Y)
    return X0X1X2Y


# In[5]:


def sigmoid(X0X1X2Y,theta):
    # calculating sigmoid function
    ita = np.dot(X0X1X2Y[:,0:3],theta)
    return 1/(1+np.exp(-ita))


# In[6]:


def newtonUpdate():
    theta = np.zeros((3,1)) # initialize theta to zeros
    X0X1X2Y = inputData()
#     oldTheta= np.ones((3,1))
#     epsilon = 1e-100
    for i in range(20):
#     while(abs(theta[0,0] - oldTheta[0,0]) > epsilon or abs(theta[1,0] - oldTheta[1,0]) > epsilon or abs(theta[2,0] - oldTheta[2,0]) > epsilon):
        #calculating the hessian matrix
        sigma = sigmoid(X0X1X2Y,theta)*(1-sigmoid(X0X1X2Y,theta))

        hessian =  np.dot(np.dot(X0X1X2Y[:,0:3].T,np.diag(sigma[:,0:1].flat)),X0X1X2Y[:,0:3])

        # gradient of log likelyhood
        Jcost =np.dot( X0X1X2Y[:,0:3].T,(sigmoid(X0X1X2Y,theta) - X0X1X2Y[:,3:4]))

         #calculating theta
        theta -= np.dot(np.linalg.inv(hessian),Jcost)
#         oldTheta = theta.copy()

    return theta





# In[7]:


def plot():
    theta= newtonUpdate()
    print(theta)
    X0X1X2Y = inputData()

    x2 = -(np.dot(X0X1X2Y[:,0:2],theta[0:2,:])/theta[2:3,:])
#     a,=plt.plot(X0X1X2Y[:50,1:2],X0X1X2Y[:50,2:3],"rx",label ="negative")
    a,=plt.plot((X0X1X2Y[np.where(X0X1X2Y[:,3]==0)])[:,1],(X0X1X2Y[np.where(X0X1X2Y[:,3]==0)])[:,2],"rx",label ="negative")

    b,=plt.plot((X0X1X2Y[np.where(X0X1X2Y[:,3]==1)])[:,1],(X0X1X2Y[np.where(X0X1X2Y[:,3]==1)])[:,2],"b^",label ="positive")
    c,=plt.plot(X0X1X2Y[:,1:2],x2 )
    plt.legend()
    plt.xlabel("Feature X1",color="r")
    plt.ylabel("Feature X2",color="r")
    plt.title("Logistic regression",color="b")
    plt.show()
    return a,b,c


# In[8]:


(a,b,c)=plot()
