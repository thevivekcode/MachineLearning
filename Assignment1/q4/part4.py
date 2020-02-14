
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
    dfX = pd.read_csv(sys.argv[1],sep="\s+",usecols=[0,1],names=['X1','X2'])
    # normalizing the data
    X1 = np.array(normalize(dfX["X1"])).reshape(-1,1)
    X2 = np.array(normalize(dfX["X2"])).reshape(-1,1)
    dfY = pd.read_csv(sys.argv[2],sep="\s+",usecols=[0],names=['Y'])

    # Alaska is represented 0 canada as 1

    Y = np.array([0 if i=="Alaska" else 1 for i in dfY["Y"]]).reshape(-1,1)
    #joining the training example as one numpy Narray
    X1X2Y = np.concatenate([X1,X2,Y],axis =1).reshape(-1,3)
    return X1X2Y


# In[5]:


def plotData(X1X2Y):
#     X1X2Y=inputData()

    #plotting the alaska data
    A1 = (X1X2Y[np.where(X1X2Y[:,2]==0)])[:,0]
    A2 = (X1X2Y[np.where(X1X2Y[:,2]==0)])[:,1]
    a,=plt.plot(A1,A2,"bX",label="Alaska")

    #plotting the canada data
    C1 = (X1X2Y[np.where(X1X2Y[:,2]==1)])[:,0]
    C2 = (X1X2Y[np.where(X1X2Y[:,2]==1)])[:,1]
    b,=plt.plot(C1,C2,"ro",label="Canada")

    #labelling the axis
    plt.xlabel("Growth Ring Diameters Fresh Water",color="r")
    plt.ylabel("Growth Ring Diameters Marine Water",color="r")
    plt.title("Data Distribution")
    plt.legend()
    plt.show(block = False)
    return a,b


# In[6]:


def cal_MU(X1X2Y):
#     X1X2Y=inputData()
    count0 = np.count_nonzero(X1X2Y[:,2] == 0)
    count1 = np.count_nonzero(X1X2Y[:,2] == 1)
    MU0 = []
    MU1 = []
    MU0.append(np.sum(X1X2Y[:,0]*(1-X1X2Y[:,2]))/count0)
    MU0.append(np.sum(X1X2Y[:,1]*(1-X1X2Y[:,2]))/count0)
    MU1.append(np.sum(X1X2Y[:,0]*X1X2Y[:,2])/count1)
    MU1.append(np.sum(X1X2Y[:,1]*X1X2Y[:,2])/count1)
    return np.array(MU0).reshape(-1,1),np.array(MU1).reshape(-1,1)


# In[7]:


def phi(X1X2Y):
#     X1X2Y=inputData()
    count1 = np.count_nonzero(X1X2Y[:,2] == 1)
    return count1/X1X2Y.shape[0]


# In[8]:


def covariance(X1X2Y,MU0,MU1 ):
#     MU0,MU1 = cal_MU()
#     X1X2Y = inputData()
    #calculating X -mu(i) for both feature vector
    X1 = np.array([i[0]-MU0[0,:] if i[2]==0 else i[0]-MU1[0,:] for i in X1X2Y])
    X2 = np.array([i[1]-MU0[1,:] if i[2]==0 else i[1]-MU1[1,:] for i in X1X2Y])
    X=np.concatenate([X1,X2],axis=1)
    return np.dot(X.T,X)/X.shape[0]



# In[9]:


def find_X2_point_linear(x1,MU0,MU1,COV,phiValue):
#     phiValue = phi()
#     (MU0,MU1) = cal_MU()
#     COV = covariance()
#     ax1+bx2+c =0
    c = -(np.dot(np.dot(MU0.T,np.linalg.inv(COV)),MU0) - np.dot(np.dot(MU1.T,np.linalg.inv(COV)),MU1))+ math.log(phiValue/(1-phiValue))
    ab =  np.dot(MU1.T,np.linalg.inv(COV)) - np.dot(MU0.T,np.linalg.inv(COV))
    a = ab[:,0]
    b = ab[:,1]
    x2 = -(c +a*x1)/b
    return x2


# In[10]:


# def LinearBoundry(X1X2Y):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
# #     X1X2Y=inputData()

#     #plotting the alaska data
#     A1 = (X1X2Y[np.where(X1X2Y[:,2]==0)])[:,0]
#     A2 = (X1X2Y[np.where(X1X2Y[:,2]==0)])[:,1]
#     a,=plt.plot(A1,A2,"bX",label="Alaska")

#     #plotting the canada data
#     C1 = (X1X2Y[np.where(X1X2Y[:,2]==1)])[:,0]
#     C2 = (X1X2Y[np.where(X1X2Y[:,2]==1)])[:,1]
#     b,=plt.plot(C1,C2,"ro",label="Canada")

#     #plotting hypothesis

#     c, = plt.plot(X1X2Y[:,0],np.array([find_X2_point_linear(i[0]) for i in X1X2Y]).reshape(-1,1),"g",label = "Decision Boundary")

#     #labelling the axis
#     plt.xlabel("X1 feature",color="r")
#     plt.ylabel("X2 feature",color="r")
#     plt.title("GDA Classification")
#     plt.legend()
#     plt.show(block = False)
#     return a,b,c


# In[11]:


# (c,d)=plotData()
# (a,b,c)=LinearBoundry()


# In[12]:


def differCovariance(X1X2Y,MU0,MU1):
#     MU0,MU1 = cal_MU()
#     X1X2Y = inputData()
    count1 = np.count_nonzero(X1X2Y[:,2] == 1)
    count0 = np.count_nonzero(X1X2Y[:,2] == 0)

    C00 = ((X1X2Y[np.where(X1X2Y[:,2]==0)])[:,0] - MU0[0,:]).reshape(-1,1)
    C01 = ((X1X2Y[np.where(X1X2Y[:,2]==0)])[:,1] - MU0[1,:]).reshape(-1,1)

    C10 = ((X1X2Y[np.where(X1X2Y[:,2]==1)])[:,0] - MU1[0,:]).reshape(-1,1)
    C11 = ((X1X2Y[np.where(X1X2Y[:,2]==1)])[:,1] - MU1[1,:]).reshape(-1,1)


    C0  = np.concatenate([C00,C01],axis=1)
    C1  = np.concatenate([C10,C11],axis=1)
    sigma0 = np.dot(C0.T,C0)/count0
    sigma1 = np.dot(C1.T,C1)/count1
    return sigma0,sigma1


# In[13]:


# (sigma0,sigma1)=differCovariance()
# print(sigma0)
# print(sigma1)


# In[14]:


def find_X2_point_quadratic(x1,MU0,MU1,sigma0,sigma1,phiValue):
#     phiValue = phi()
#     (MU0,MU1) = cal_MU()
#     (sigma0,sigma1)=differCovariance()
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    a0 = sigma0_inv[0,0]
    b0 = sigma0_inv[1,1]
    c0 = sigma0_inv[0,1]
    a1 = sigma1_inv[0,0]
    b1 = sigma1_inv[1,1]
    c1 = sigma1_inv[0,1]

    p1q1 = np.dot(MU1.T, sigma1_inv)
    p0q0 = np.dot(MU0.T, sigma0_inv)

    p1 =p1q1[0,0]
    q1= p1q1[0,1]
    p0 =p0q0[0,0]
    q0= p0q0[0,1]

    '''

    representaation assumption to solve the quadratic equation

    sigma0_inv = a0  c0       sigma1_inv = a1  c1
                 c0  b0                    c1  b1

    np.dot(MU1.T sigma1_inv) = p1  q1
    np.dot(MU0.T sigma0_inv) = p0  q0



    A * x2^2 + B * X2 + C = 0

    '''

    sigma0_det = np.linalg.det(sigma0)
    sigma1_det = np.linalg.det(sigma1)




    A = ((b0 - b1)/2)

    B = x1*(c0 - c1) + (q1 -q0)

    C = (x1**2)*((a0-a1)/2) + x1*(p1-p0) - ( np.dot(np.dot(MU1.T,sigma1_inv),MU1) - np.dot(np.dot(MU0.T,sigma0_inv),MU0) )/2  + math.log(phiValue/(1-phiValue)) + math.log(math.sqrt(abs(sigma0_det))/math.sqrt(abs(sigma1_det)))
#     print(C)
    D =(B**2 - 4*A*C[0,0])
    x2_0 = (-B + np.sqrt(D))/(2*A)
    x2_1 = (-B - np.sqrt(D))/(2*A)

    return x2_0, x2_1




# In[15]:


# find_X2_point_quadratic(2)


# In[16]:


def boundry(X1X2Y,MU0,MU1,sigma0,sigma1,phiValue,COV):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
#     X1X2Y=inputData()

    #plotting the alaska data
    A1 = (X1X2Y[np.where(X1X2Y[:,2]==0)])[:,0]
    A2 = (X1X2Y[np.where(X1X2Y[:,2]==0)])[:,1]
    a,=plt.plot(A1,A2,"bX",label="Alaska")

    #plotting the canada data
    C1 = (X1X2Y[np.where(X1X2Y[:,2]==1)])[:,0]
    C2 = (X1X2Y[np.where(X1X2Y[:,2]==1)])[:,1]
    b,=plt.plot(C1,C2,"ro",label="Canada")

    #plotting hypothesis
    x1  = np.linspace(-2,2,10)
    Pair=[find_X2_point_quadratic(i,MU0,MU1,sigma0,sigma1,phiValue) for i in x1]
    q, = plt.plot(x1,[i[1] for i in Pair],"g",label = "Quardtic Boundary")
    l, = plt.plot(X1X2Y[:,0],np.array([find_X2_point_linear(i[0],MU0,MU1,COV,phiValue) for i in X1X2Y]).reshape(-1,1),"orange",label = "Linear Boundary")


    #labelling the axis
    plt.xlabel("Growth Ring Diameters Fresh Water",color="r")
    plt.ylabel("Growth Ring Diameters Marine Water",color="r")
    plt.title("GDA Classification")
    plt.legend()
    plt.show()
    return a,b,l,q


# In[17]:


# quadratic_boundary()


# In[18]:


def main():
    X1X2Y = inputData()
    (MU0,MU1) = cal_MU(X1X2Y)
    (sigma0,sigma1) = differCovariance(X1X2Y,MU0,MU1)
    phiValue =phi(X1X2Y)
    COV = covariance(X1X2Y,MU0,MU1 )
    print("---------MU values----------")
    print("MU0")
    print(MU0)
    print("MU1")
    print(MU1)
    print("------------------------------")
    print("------SAME COVARIANCE MATRIX---")
    print(COV)
    print("-------------------------------")
    print("------DIFFERENT COVARAINCE MATRIX-------")
    print("COV0")
    print(sigma0)
    print("COV1")
    print(sigma1)
    print("----------------------------------------")
    # plotData(X1X2Y)
    boundry(X1X2Y,MU0,MU1,sigma0,sigma1,phiValue,COV)

# In[19]:


main()
