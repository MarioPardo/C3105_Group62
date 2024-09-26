#COMP3105  Assignment 1  Group 62


import os
import autograd.numpy as np  # when testing gradient
from cvxopt import matrix, solvers
import pandas as pd



##Q1#######

#a
def minimizeL2(X,y):
    #from lecture, we know the analytic solution to L2 loss
    # is   w = ((XtX)^-1)Xty

    x_trans = X.T

    #(XtX)^-1
    firstpart = np.linalg.inv(x_trans @ X)
    #secondpart XtY
    secpart = np.dot(x_trans, y)

    return (firstpart @ secpart)



#b

def minimizeL1(X,y):

    n, d = X.shape
    Identity = np.eye(n)
    nOnes = np.ones(n)
    objfunc = np.concatenate([np.zeros(d), nOnes])  # [0's for w, 1's for delta]

    #constraints
    con1 = np.hstack([X, -Identity])  #Xw - y <= delta
    con2 = np.hstack([-X, -Identity])  #y - Xw <= delta
    constraints = np.vstack([con1, con2]) #combine to make the constraint array

    #y portion of |Xw-y| < delta
    h1 = y   #Xw - delta <= y
    h2 = -y   #y-Xw - delta <= -y
    h = np.concatenate([h1,h2])
    h = h.reshape(-1, 1) #turn to col vector
    h = matrix(h, tc='d')  # Ensure h is a cvxopt matrix in double precision

    constraints_cvxopt = matrix(constraints)
    h_cvxopt = matrix(h)
    objfunc_cvxopt = matrix(objfunc)

    solvers.options['show_progress'] = False  # Silence solver output
    solution = solvers.lp(objfunc_cvxopt, constraints_cvxopt, h_cvxopt)

    return np.array(solution['x'])[:d]



#c

def minimizeLinf(X, y):
    
    n, d = X.shape  # n: number of data points, d: number of features
    objfunc = np.concatenate([np.zeros(d), [1]])  # Objective: [0's for w, 1 for delta]
    Identity = np.eye(n)

    
    con1 = np.hstack([X, -np.ones((n, 1))])   # Xw - delta <= y
    con2 = np.hstack([-X, -np.ones((n, 1))])  # y - Xw <= delta
    constraints = np.vstack([con1, con2])

    # y portion of |Xw - y| <= delta
    h1 = y  # Xw - y <= delta)
    h2 = -y  # y - Xw <= delta)
    h = np.concatenate([h1, h2])

    constraints_cvxopt = matrix(constraints, tc='d')
    h_cvxopt = matrix(h, tc='d')
    objfunc_cvxopt = matrix(objfunc, tc='d')

    solvers.options['show_progress'] = False
    solution = solvers.lp(objfunc_cvxopt, constraints_cvxopt, h_cvxopt)

    # Return the optimized weights (w), which are the first d elements of the solution
    return np.array(solution['x'])[:d]




##Q2   


#a
def linearRegL2Obj(w, X, y):

    pred = X @ w
    diff = pred = y
    objectiveVal = (1/2) * np.mean(diff ** 2)
    n = X.shape[0]
    gradient = (X.T @ diff) / n

    return objectiveVal, gradient

#b
def gd(func, w_init, X, y, step_size, max_iter, tol=1e-10):

    w = w_init

    for i in range(max_iter):
        objval, gradient = func(w,X,y)
        
        norm = np.linalg.norm(gradient)
        if norm < tol: #stop when gradient is good enough
            break;

        w = w - step_size * gradient

    return w

#Q2c
def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def logisticRegObj(w, X, y):
    pred = X @ w
    n = X.shape[0]
    
    objval = (1/n) * np.sum(np.logaddexp(0, -pred) - y * pred)
    grad = (1/n) * X.T @ (sigmoid(pred) - y)
    
    return objval, grad