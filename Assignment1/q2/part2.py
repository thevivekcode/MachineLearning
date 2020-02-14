
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


def inputData():
    df = pd.read_csv(sys.argv[1])
    X_0 = np.ones((len(df),1))
    X1X2Y = df.to_numpy()
    X0X1X2Y = np.append(X_0,X1X2Y, axis=1)
    np.random.shuffle(X0X1X2Y)  #shuffling data to make it random for better distribution
    return X0X1X2Y




# In[4]:


def sampleData():
    Stheta = np.reshape(np.array([3,1,2]),(3,1))
    SX_0 = np.ones((1000000))
    SX_1 = np.random.normal(size = 1000000, loc = 3, scale = 2) # loc = mean scale = standard deviation
    SX_2 = np.random.normal(size = 1000000, loc = -1, scale = 2)
    SX = np.reshape(np.array([SX_0,SX_1,SX_2]).T,(-1,3) ) # creating design matrix of sample data
    hypoT = np.dot(SX,Stheta)

    SY = np.reshape(np.random.normal(size = 1000000, loc = hypoT.T , scale = math.sqrt(2)),(-1,1))
    SX0X1X2Y = np.append(SX,SY,axis=1)

    np.random.shuffle(SX0X1X2Y)

#     plot to visualize the generated sample

#     plt.plot(SX0X1X2Y[:,1:2],SX0X1X2Y[:,3:4],"bo",label ="X1 feature")
#     plt.plot(SX0X1X2Y[:,2:3],SX0X1X2Y[:,3:4],"rX",label ="X2 feature")
#     plt.xlabel("Input Feature",color="red")
#     plt.ylabel("Target value",color="red")
#     plt.title("Sampling Million Data")
#     plt.legend()
#     plt.show()

    return SX0X1X2Y


# In[5]:


def costFuction(SX0X1X2Y,theta):
    hypothesis = np.dot(SX0X1X2Y[:,0:3],np.array(theta))
    hypothesis = np.reshape(hypothesis,(-1,1))
    error = SX0X1X2Y[:,3:4]-hypothesis
    errorSquared =error**2
    examples = SX0X1X2Y.shape[0] #X.shape[0] represents number of training example
    return np.sum(errorSquared)/(2*examples)


# In[6]:


# a = sampleData()
# print(a)
# costFuction(a,[3,1,2])


# In[7]:


def gradientFunction(SX0X1X2Y,theta):
    error = np.reshape(np.reshape(np.dot(SX0X1X2Y[:,0:3],np.array(theta)),(-1,1)) - SX0X1X2Y[:,3:4],(-1,1))  # h(X) -y
    grad_cost = np.zeros((3,1))
    grad_cost[0] = np.sum(error*SX0X1X2Y[:,0:1])/(SX0X1X2Y.shape[0])
    grad_cost[1] = np.sum(error*SX0X1X2Y[:,1:2])/(SX0X1X2Y.shape[0])
    grad_cost[2] = np.sum(error*SX0X1X2Y[:,2:3])/(SX0X1X2Y.shape[0])
    return grad_cost


# In[8]:


# SX0X1X2Y = sampleData()
# gradientFunction(SX0X1X2Y,[3,1,2])


# In[9]:


# implement Stochastic gradient descent unitil it converges

def stochasticGradientDescent(SX0X1X2Y,learningRate,batchSize):



    theta = np.zeros((3,1))  # initializing theta vector

    Batch_SX0X1X2Y = np.reshape(SX0X1X2Y,(-1,batchSize,4))

    stopping_criteria  = 0
    avgIterations = 0
    if batchSize < 10001:
        stopping_criteria = 1e-4
        avgIterations = 2000
    else:
        stopping_criteria = 1e-3
        avgIterations = 1000

    print("------------------------------------------------")
    print("Stopping_criteria is  {}".format(stopping_criteria))
    print("AvgIterations for convergence test is:  {}".format(avgIterations))
    print("-------------------------------------------------")
    threshold = 1000000
    iterations = 0
    count = 0
    costNew = 0
    cost = 0
    thetaList = [] # stroing theta in list to for plotting purpose
    while(True):

        for i in Batch_SX0X1X2Y:
            iterations += 1
            thetaList.append(theta.copy())
            cost += costFuction(i,theta)
            count += 1

            if count == avgIterations:

                costOld = costNew
                costNew = cost/avgIterations
                cost = 0
                count = 0
#                 print( abs(costOld - costNew))
                if abs(costOld - costNew) <= stopping_criteria:
                    return theta,iterations,thetaList


            if iterations > threshold:
                return theta,iterations,thetaList

            theta -= learningRate * gradientFunction(i,theta)





# In[10]:


def plot(theta,iterations,thetaList):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set(xlim=(0, 4), ylim=(0, 2), zlim = (0,3))
#     print(thetaList)
    thetaList0 = thetaList[:,0].tolist()
    thetaList1 = thetaList[:,1].tolist()
    thetaList2 = thetaList[:,2].tolist()


    lines, = ax.plot([], [],[],color = 'green',markersize = 2)
    def animationLine(i):
            lines.set_data(thetaList0[:i],thetaList1[:i])
            lines.set_3d_properties(thetaList2[:i])
            lines.set_marker("x")
            return lines,
    anim = animation.FuncAnimation(fig, animationLine,frames=len(thetaList0),interval = 1,repeat=False)
    plt.xlabel("theta 0",color ='r')
    plt.ylabel("theta 1",color ='r')
    ax.set_zlabel("theta 2",color ='r')
    plt.title("Movement of parameters")
    plt.show()
    return anim


# In[11]:


def main():
    SX0X1X2Y = sampleData()
    learningRate = 1e-3
    batchSize = int(sys.argv[2])
    print("--------------------------------------------------")
    print("Learning  Rate is: {}".format(learningRate))
    print("Batch Size is: {}".format(batchSize))
    print("--------------------------------------------------")
    startTime = time()
    (theta,iterations,thetaList) = stochasticGradientDescent(SX0X1X2Y,learningRate,batchSize)
    endTime  =time() - startTime
    print("--------------------------------------------------")
    print("theta is {}".format(theta.tolist()))
    print("Total iterations {}".format(iterations))
    print("Total time took: {}".format(endTime))
    print("--------------------------------------------------")
    thetaList=np.array(thetaList).reshape(-1,3)
    plotting = plot(theta,iterations,thetaList)
    print("--------------------------------------------------")
    print("Testing Data")

    X0X1X2Y=inputData()
    testError = costFuction(X0X1X2Y,theta)
    print("Test error is {}".format(testError))
    print("--------------------------------------------------")
    return plotting


# In[12]:


main()


# In[13]:


X0X1X2Y=inputData()
theta =[3,1,2]
testError = costFuction(X0X1X2Y,theta)
print(testError)
