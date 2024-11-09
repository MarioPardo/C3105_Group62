import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

##Q1

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
    indmax = np.argmax(WX, axis=1)

    result = convertToOneHot(indmax, WX.shape[1])

    return result

#Q1c

def calculateAcc(Yhat, Y):
    return np.mean(np.all(Yhat == Y,axis=1))





##