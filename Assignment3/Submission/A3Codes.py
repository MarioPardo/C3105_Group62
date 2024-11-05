import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

##Q1

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

#Q1a

def multinomial_deviance_loss(W, X, Y, d, k):

    W = W.reshape(d, k)
    WX = np.dot(X, W)

    log_sum_exp = logsumexp(WX, axis=1)
    y_WX = np.sum(Y * WX, axis=1)
    
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
    #print("WX",WX)
    indmax = np.argmax(WX, axis=1)
    #print("indmax",indmax)

    result = convertToOneHot(indmax, WX.shape[1])
    #print("result",result)

    return result

#Q1c

def calculateAcc(Yhat, Y):
    #print("Prediction", Yhat)
    #print("Actual", Y)
    return np.mean(Yhat == Y)





##