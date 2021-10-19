# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path='D:\\Programs\\classification_data.txt'

data=pd.read_csv(path,header=None, names=('Exam 1','Exam 2','Admitted'))

print(data.head(10))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

positive=data[data['Admitted'].isin([1])]
negative=data[data['Admitted'].isin([0])]

print("positive is \n", positive.head(10),'\n')

print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

print("positive is \n", negative.head(10))

print("++++++++++++++++++++++++++++++++++++++++++++++++++++")

fig,ax=plt.subplots(figsize=(8,5))
ax.scatter(positive['Exam 1'],positive['Exam 2'],s=50 , c='r' ,marker='o',label='Admitted')

ax.scatter(negative['Exam 1'],negative['Exam 2'],s=60 , c='b' ,marker='x',label='Not Admitted')

ax.legend()
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
#nums=np.arange(-10,10)
def sigmoid(z):
     return 1/(1+np.exp(-z))

nums=np.arange(-10,10)

fig ,ax =plt.subplots(figsize=(8,5))
ax.plot(nums,sigmoid(nums),'r')



data.insert(0,'ones',1)

col=data.shape[1] #100x4
x=data.iloc[:,0:col-1]
y=data.iloc[:,col-1:col]

#print(x,'\n')
#print("++++++++++++++++++++++++++++++++++++++++++++++++++++")
#print(y.head(10))
x=np.array(x.values)
y=np.array(y.values)
theta=np.zeros(3)
theta=np.matrix(theta)


def cost(theta,x,y):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)
    first=np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second=np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))
    return np.sum(first-second)/(len(x))
thiscost=cost(theta,x,y)
#print('cost =', thiscost ) 

def gradient(theta,x,y):
    theta=np.matrix(theta)
    x=np.matrix(x)
    y=np.matrix(y)
    
    parameters=int(theta.ravel().shape[1])
    grad=np.zeros(parameters)
    
    error=sigmoid(x*theta.T)-y
    for i in range(parameters):
        term=np.multiply(error,x[:,i])
        grad[i]=np.sum(term)/len(x)
    
    return grad
import scipy.optimize as opt

result=opt.fmin_tnc(func=cost,x0=theta,fprime=gradient,args=(x,y))

costafteroptimize=cost(result[0],x,y)

def predict(theta,x):
    prob=sigmoid(x*theta.T)
    return(1 if x>=0.5 else 0 for x in prob)

theta_min = np.matrix(result[0])
predictions=predict(theta_min,x)

correct=[1 if ((a==1 and b==1)or(a==0 and b==0))else 0 for (a,b)in 
         zip(predictions,y) ]

accuracy = (sum(map(int,correct))%len(correct))

print('accuracy={0}%'.format(accuracy))

 