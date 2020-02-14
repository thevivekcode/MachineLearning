
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.animation as animation
import sys


# In[2]:




# In[3]:


def inputData():
    dfX = pd.read_csv(sys.argv[1],usecols=[0],names=["X"],header=None)
    inputListX=dfX['X'].tolist()
    dfY = pd.read_csv(sys.argv[2],usecols=[0],names=["Y"],header=None)
    inputListY=dfY['Y'].tolist()
    return inputListX,inputListY




# In[4]:


#normalize function (x-mean)/std for each value in input listX

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    data = data -mean
    data = data/std
    return data



# In[5]:


# step 1: creating input DATA
#         Making Design matrix "X" with X0=1

def align():

    (inputListX,inputListY) = inputData()
    numpyListX = normalize(np.reshape(np.array(inputListX),(-1,1)))
    numpyListY = np.reshape(np.array(inputListY),(-1,1))

    data = np.append( numpyListX,numpyListY, axis = 1)

    np.random.shuffle(data)  #shuffling data to make it random for better distribution

    x = data[:, 0 : 1]  #copy all rows and only copy zeroth col [0,1)
    y = data[:, 1 : 2]  #copy all rows and only copy first col [1,2)
    ones = np.ones((x.shape[0], 1))
    X = np.append(ones, x, axis = 1)
    return x,y,X


# In[6]:


# step 3: define a cost function

def costFuction(y,X,theta):
    hypothesis = np.dot(X,np.array(theta))
    hypothesis = np.reshape(hypothesis,(100,1))
    error = y-hypothesis
    errorSquared =error**2
    examples = X.shape[0] #X.shape[0] represents number of training example
    return np.sum(errorSquared)/(2*examples)





# In[7]:


#step 4: define a gradient function

def gradientFunction(X,y,theta):
    error = np.reshape(np.dot(X,theta),(-1,1)) - y  # h(X) -y
    np.reshape(error,(-1,1))
    grad_cost = np.zeros((2,1))
    grad_cost[0] = np.sum(error*X[:,0:1])/(X.shape[0])
    grad_cost[1] = np.sum(error*X[:,1:2])/(X.shape[0])
    return grad_cost



# In[8]:


# implement gradient descent unitil it converges

def gradientDescent(x,y,X,learningRate ):

#     (x,y,X) = align()

    theta = np.zeros((X.shape[1],1))  # initializing theta vector




    costPlot = [] # list for plot of costfunction
    thetaList0 =[]
    thetaList1 =[]

    #first iteration to implement do while
    oldTheta = theta.copy()
    thetaList0 = np.append(thetaList0,theta[0:1,:])
    thetaList1 = np.append(thetaList1,theta[1:2,:])

    theta -= learningRate*gradientFunction(X,y,theta) #theta(t+1)
    thetaList0 = np.append(thetaList0,theta[0:1,:])
    thetaList1 = np.append(thetaList1,theta[1:2,:])
    costOld = costFuction(y,X,oldTheta)
    costNew = costFuction(y,X,theta)
    costPlot.append(costOld)
    costPlot.append(costNew)

    while(abs(costNew - costOld) > 1e-15 ):
        costOld = costNew.copy()
        oldTheta = theta.copy()
        ''''''
        theta -= learningRate*gradientFunction(X,y,theta) #theta(t+1)
        ''''''
        costNew = costFuction(y,X,theta)
        thetaList0 = np.append(thetaList0,theta[0:1,:])
        thetaList1 = np.append(thetaList1,theta[1:2,:])
        costPlot.append(costNew)
    print("part A")
    print("-----------final values of theta is -----------")
    print(theta)
    print("-----------------------------------------------")
    return x,y,X,theta,costPlot,thetaList0,thetaList1


# In[9]:


def plot(x,y,X,theta,costPlot,thetaList0,thetaList1):

    # plot data point and hypothesis
    plt.plot(x,y,"bp",label="Data points")
    hypo, = plt.plot(x,theta[0]+x*theta[1] , color='r',label="hypothesis")
    plt.title("linear regression" ,color = 'b')
    plt.xlabel("Acidity of Wine",color= 'r')
    plt.ylabel("Density of Wine",color='r')
    #plot cost function
#     twoDplot, = plt.plot(costPlot)
    plt.legend()
    plt.show(block = False)
    return  hypo


# In[10]:


# plot()


# In[11]:


def surface_plot(x,y,X,theta,costPlot,thetaList0,thetaList1):


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #create 100 different theta values for plotting plot_surface
    th0 = np.linspace(theta[0:1,:]-1,theta[0:1,:]+1,100)
    th1 = np.linspace(theta[1:2,:]-1,theta[1:2,:]+1,100)


    # meshGrid to create all possible pairs od theta in X and Y axis
    theta_0,theta_1 = np.meshgrid(th0,th1)

    # calculating CostFun for all pairs that we got from above meshGrid
    CostFun=[]
    for i in range(100):
        for j in range(100):
            theta=[]
            theta.append(theta_0[i,j])
            theta.append(theta_1[i,j])
            CostFun.append(costFuction(y,X,theta))

    CostFun= np.reshape(CostFun,theta_0.shape)
    ax.plot_surface(theta_0,theta_1,CostFun,alpha = 1,cmap = cm.GnBu)
    ax.set_xlabel('theta 0',color ='r')
    ax.set_ylabel('theta 1',color ='r')
    ax.set_zlabel('Loss-Function',color ='r')

    #initializs the line for 3d animation
    lines, = ax.plot([], [],[],color = 'green',markersize = 2)
    def animationLine(i):
            lines.set_data(thetaList0[:i],thetaList1[:i])
            lines.set_3d_properties(costPlot[:i])
            lines.set_marker("x")
            return lines,


    anim = animation.FuncAnimation(fig, animationLine,frames=len(thetaList0),interval=200, repeat = True)
    plt.title("3D Loss-Function" , color ="b")
    plt.show(block = False)
    return anim


# In[12]:


# surface_plot()


# In[13]:


def countourPLot(x,y,X,theta,costPlot,thetaList0,thetaList1):
    #contour plot
#     (x,y,X) = align()
#     ( x,y,X,theta,costPlot,thetaList0,thetaList1) = gradientDescent(x,y,X)

    fig = plt.figure()
    ax = fig.add_subplot(111)

  #create 100 different theta values for plotting plot_surface
    th0 = np.linspace(theta[0:1,:]-1,theta[0:1,:]+1,100)
    th1 = np.linspace(theta[1:2,:]-1,theta[1:2,:]+1,100)

#     (x,y,X) = align()
    # meshGrid to create all possible pairs od theta in X and Y axis
    theta_0,theta_1 = np.meshgrid(th0,th1)

    CostFun=[]
    for i in range(100):
        for j in range(100):
            theta=[]
            theta.append(theta_0[i,j])
            theta.append(theta_1[i,j])
            CostFun.append(costFuction(y,X,theta))

    CostFun= np.reshape(CostFun,theta_0.shape)
    CS = ax.contour(theta_0,theta_1,CostFun,alpha = 1)
    ax.clabel(CS, inline=1, fontsize=10)
    ax.set_xlabel('theta 0',color ='r')
    ax.set_ylabel('theta 1',color ='r')

    lines, = ax.plot([], [],color = 'green',markersize = 2)
    def animationLine(i):
            lines.set_data(thetaList0[:i],thetaList1[:i])
            lines.set_marker("x")
            return lines,


    anim = animation.FuncAnimation(fig, animationLine,frames=len(thetaList0),interval=200, repeat = False)
    plt.title("Loss-Function Contour",color="b")
    plt.show()
    return anim


# In[14]:


# countourPLot()


# In[15]:


def main():
    startTime = time()
    #taking input and converting it into apropriate formats to be used further
    (x,y,X) = align()
    learningRate = float(sys.argv[3])
    #calculate gradient descent for linear regression
    (x,y,X,theta,costPlot,thetaList0,thetaList1) = gradientDescent(x,y,X,learningRate)

    # plot 2D graph and learned hypothesis
    hypo = plot(x,y,X,theta,costPlot,thetaList0,thetaList1)


    # plot surface_plot of the cost function and animate the hypothesis
    anim1 = surface_plot(x,y,X,theta,costPlot,thetaList0,thetaList1)

    # plot countour and animate the hypothesis
    anim2 = countourPLot(x,y,X,theta,costPlot,thetaList0,thetaList1)
    endTime = time() - startTime
    print("-----time taken to finsih is-----" )
    print(endTime)
    print("---------------------------------")
    return hypo,anim1,anim2


# In[16]:


(a,b,c)=main()


# In[ ]:
