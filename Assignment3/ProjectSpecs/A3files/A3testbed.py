# COMP 3105 Assignment 3
# Carleton University
# NOTE: This is a sample script to show you how your functions will be called. 
#       You can use this script to visualize your models once you finish your codes. 
#       This script is not meant to be thorough (it does not call all your functions).
#       We will use a different script to test your codes. 
from matplotlib import pyplot as plt

#import YourName.A3codes as A3codes
from A3helpers import augmentX, plotModel, generateData, plotPoints, convertToOneHot
from scipy.spatial.distance import cdist

##Q1

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

#Q1a

def multinomial_deviance_loss(W, X, Y, d, k):

    W = W.reshape(d, k)
    WX = np.dot(X, W)

    log_sum_exp = logsumexp(WX, axis=1)
    y_WX = np.sum(WX*Y, axis=1)
    
    loss = np.mean(log_sum_exp - y_WX)
    
    return loss

def minMulDev(X, Y):
    n, d = X.shape
    k = Y.shape[1]
    
    #must be 1D Array as this is what minimize requires
    W_init = np.zeros(d * k)
    
    def objective(W):
        return multinomial_deviance_loss(W, X, Y, d, k)
    
    # Minimize the objective function
    result = minimize(objective, W_init, method='L-BFGS-B')
    
    return result.x.reshape(d, k)


#Q1b

def classify(X,W):
    WX = np.dot(X,W)
    indmax = np.argmax(WX, axis=1)
    
    
    return convertToOneHot(indmax, WX.shape[1])

#Q1c

def calculateAcc(Yhat, Y):
    return np.mean(np.all(Yhat == Y,axis=1))




#Q2






#Q3


#a

def kmeans(X,k,max_iter=1000):
    if(max_iter <= 0):
        return False
     
    n,d = X.shape
    U = X[np.random.choice(n,k,replace=False)]

    for i in range(max_iter):

        D = cdist(X, U, 'euclidean')
        Y = np.argmin(D, axis=1)

        old_U = U
        Y = convertToOneHot(Y, k)
        YtY = Y.T @ Y + 1e-8 * np.eye(k) #for stability
        U = np.linalg.inv(Y.T @ Y) @ Y.T @ X

        if np.allclose(U, old_U):
            break

        
    obj_val = 1/(2*n) * np.linalg.norm(X - Y @ U, 'fro')**2

    return Y, U, obj_val




##


def _plotCls():

	n = 100

	# Generate data
	Xtrain, Ytrain = generateData(n=n, gen_model=1, rand_seed=0)
	Xtrain = augmentX(Xtrain)

	# Learn and plot results
	W = minMulDev(Xtrain, Ytrain)
	print(f"Train accuaracy {calculateAcc(Ytrain, classify(Xtrain, W))}")

	plotModel(Xtrain, Ytrain, W, classify)

	return


def _plotKmeans():

	n = 100
	k = 3

	Xtrain, _ = generateData(n, gen_model=2)

	Y, U, obj_val = kmeans(Xtrain, k)
	plotPoints(Xtrain, Y)
	plt.legend()
	plt.show()

	return


if __name__ == "__main__":

	_plotCls()
	_plotKmeans()
