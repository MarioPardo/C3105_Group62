# The functions in this cell are the answers to the assignment
import os

import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import minimize
import pandas as pd
from A2helpers import generateData, plotModel, linearKernel, polyKernel, gaussKernel, plotAdjModel, plotDualModel

solvers.options['show_progress'] = False

# Q1a
def minBinDev(X, y, lamb):
    _, d = X.shape
    def obj_fun(u):
        w0 = u[-1]
        w = u[:-1]
        w = w[:, None]
        
        loss = np.sum(np.logaddexp(0, -y * (X @ w + w0)))
        reg = 0.5 * lamb * float(w.T @ w)
        return loss + reg
            
    res = minimize(obj_fun, np.ones(d + 1))
    w = res.x[:-1][:, None]
    w0 = res.x[-1]
    return w, w0

# Q1b
def minHinge(X, y, lamb, stablizer=1e-5):
    n, d = X.shape
    
    q = np.concatenate([np.zeros(d + 1), np.ones(n)])
    q = matrix(q)
    
    G = np.concatenate([np.concatenate([np.zeros_like(X), np.zeros([n, 1]), -np.eye(n)], axis=1),
                        np.concatenate([-y *X, -y*np.ones([n, 1]), -np.eye(n)], axis=1)])
    G = matrix(G)
    
    h = np.concatenate([np.zeros(n), -np.ones(n)])
    h = matrix(h)
    
    P = np.zeros([d + 1 + n, d + 1 + n])
    P[:d, :d] = np.eye(d) * lamb # w part
    P = matrix(P + np.eye(d + 1 + n) * stablizer)
    
    res = solvers.qp(P, q, G, h)
    if res['status'] != 'optimal' and res['status'] != 'unknown':  # for our own debug purpose
        raise ValueError("Something's wrong with cvxopt solver")
    w = np.array(res['x'][:d])
    w0 = res['x'][d]
    return w, w0
    
# Q1c
def classify(Xtest, w, w0):
    return np.sign(Xtest @ w + w0)


# Q1d
def synExperimentsRegularize():
    n_runs = 100
    n_train = 100
    n_test = 1000
    lamb_list = [0.001, 0.01, 0.1, 1.]
    gen_model_list = [1, 2, 3]
    train_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(lamb_list), len(gen_model_list), n_runs])
    for r in range(n_runs):
        for i, lamb in enumerate(lamb_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)
                
                w, w0 = minBinDev(Xtrain, ytrain, lamb)
                train_acc_bindev[i, j, r] = np.mean(classify(Xtrain, w, w0) == ytrain)
                test_acc_bindev[i, j, r] = np.mean(classify(Xtest, w, w0) == ytest)
                
                w, w0 = minHinge(Xtrain, ytrain, lamb)
                train_acc_hinge[i, j, r] = np.mean(classify(Xtrain, w, w0) == ytrain)
                test_acc_hinge[i, j, r] = np.mean(classify(Xtest, w, w0) == ytest)
                
    # TODO: compute the average accuracies over runs
    # TODO: combine accuracies (bindev and hinge)
    # TODO: return 4-by-6 train accuracy and 4-by-6 test accuracy
    train_acc = np.concatenate([np.mean(train_acc_bindev, axis=2), np.mean(train_acc_hinge, axis=2)], axis=1)
    test_acc = np.concatenate([np.mean(test_acc_bindev, axis=2), np.mean(test_acc_hinge, axis=2)], axis=1)    
    return train_acc, test_acc

print("Q1d")
train, test = synExperimentsRegularize()
print(train)
print(test)
    
# Q2a
def adjBinDev(X, y, lamb, kernal_func):
    n, _ = X.shape
    def dev(x):
        a0 = x[-1]
        a = x[:-1]
        a = a[:, None]
        K = kernal_func(X, X)
        
        loss = np.sum(np.logaddexp(0, -y * (K @ a + a0)))
        reg = 0.5 * lamb * float(a.T @ K @ a)
        return loss + reg
    
    res = minimize(dev, np.ones(n + 1))
    a = res.x[:-1][:, None]
    a0 = res.x[-1]
    return a, a0

# Q2b
def adjHinge(X, y, lamb, kernal_func, stablizer=1e-5):
    n, _ = X.shape
    K = kernal_func(X, X)
    
    q = np.concatenate([np.zeros(n + 1), np.ones(n)])
    q = matrix(q)
    
    G = np.concatenate([np.concatenate([np.zeros([n, n]), np.zeros([n, 1]), -np.eye(n)], axis=1),
                        np.concatenate([-y * K, -y*np.ones([n, 1]), -np.eye(n)], axis=1)])
    G = matrix(G)
    
    h = np.concatenate([np.zeros(n), -np.ones(n)])
    h = matrix(h)
    
    P = np.zeros([n + 1 + n, n + 1 + n])
    P[:n, :n] = K * lamb
    P = matrix(P + np.eye(n + 1 + n) * stablizer)
    
    res = solvers.qp(P, q, G, h)
    
    a = np.array(res['x'][:n])
    a0 = res['x'][n]
    return a, a0
    
# Q2c
def adjClassify(Xtest, a, a0, X, kernal_func):
    return np.sign(kernal_func(Xtest, X) @ a + a0)

# Q2d
def synExperimentsKernel():
    n_runs = 10
    n_train = 100
    n_test = 1000
    lamb = 0.001
    kernel_list = [linearKernel,
                    lambda X1, X2: polyKernel(X1, X2, 2),
                    lambda X1, X2: polyKernel(X1, X2, 3),
                    lambda X1, X2: gaussKernel(X1, X2, 1.0),
                    lambda X1, X2: gaussKernel(X1, X2, 0.5)]
    gen_model_list = [1, 2, 3]
    train_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_bindev = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    train_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    test_acc_hinge = np.zeros([len(kernel_list), len(gen_model_list), n_runs])
    for r in range(n_runs):
        for i, kernel in enumerate(kernel_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, ytest = generateData(n=n_test, gen_model=gen_model)
                
                a, a0 = adjBinDev(Xtrain, ytrain, lamb, kernel)
                train_acc_bindev[i, j, r] = np.mean(adjClassify(Xtrain, a, a0, Xtrain, kernel) == ytrain)
                test_acc_bindev[i, j, r] = np.mean(adjClassify(Xtest, a, a0, Xtrain, kernel) == ytest)
                
                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_acc_hinge[i, j, r] = np.mean(adjClassify(Xtrain, a, a0, Xtrain, kernel) == ytrain)
                test_acc_hinge[i, j, r] = np.mean(adjClassify(Xtest, a, a0, Xtrain, kernel) == ytest)
                
    # TODO: compute the average accuracies over runs
    # TODO: combine accuracies (bindev and hinge)
    # TODO: return 5-by-6 train accuracy and 5-by-6 test accuracy
    train_acc = np.concatenate([np.mean(train_acc_bindev, axis=2), np.mean(train_acc_hinge, axis=2)], axis=1)
    test_acc = np.concatenate([np.mean(test_acc_bindev, axis=2), np.mean(test_acc_hinge, axis=2)], axis=1)
    return train_acc, test_acc

print("Q2d")
train, test = synExperimentsKernel()
print(train)
print(test)
    
# Q3a
def dualHinge(X, y, lamb, kernal_func, stablizer=1e-5):
    n, _ = X.shape
    K = kernal_func(X, X)
    deltaY = np.diag(y.flatten())
    
    q = -np.ones(n)
    q = matrix(q)
    
    G = np.concatenate([-np.eye(n), np.eye(n)])
    G = matrix(G)
    
    h = np.concatenate([np.zeros(n), np.ones(n)])
    h = matrix(h)
    
    P = (1/lamb) * deltaY * K * deltaY
    P = matrix(P + np.eye(n) * stablizer)
    
    A = matrix(y.T.astype(float))
    b = matrix(0.)
    
    res = solvers.qp(P, q, G, h, A, b)
    a = np.array(res['x'][:n])
    
    index = np.abs(a-0.5).argmin()
    y_i = y[index]
    k_i = K.T[index]
    
    b = y_i - ((1/lamb) * k_i @ deltaY @ a)
    return a, b
    
# Q3b
def dualClassify(Xtest, a, b, X, y, lamb, kernal_func):
    return np.sign(1/lamb * np.dot(kernal_func(Xtest, X), np.dot(np.diag(y.flatten()), a)) + b)

#Q3c
def cvMnist(dataset_folder, lamb_list, kernel_list, k=5):
    train_data = pd.read_csv(f"{dataset_folder}/A2train.csv", header=None).to_numpy()
    X = train_data[:, 1:] / 255.
    y = train_data[:, 0][:, None]
    y[y == 4] = -1
    y[y == 9] = 1
    
    cv_acc = np.zeros([k, len(lamb_list), len(kernel_list)])
    
    # TODO: perform any necessary setup   
    k_fold_indices = np.array_split(np.random.permutation(X.shape[0]), k)
    
    for i, lamb in enumerate(lamb_list):
        for j, kernel_func in enumerate(kernel_list):
            for l in range(k):
                Xtrain = X[np.concatenate([k_fold_indices[m] for m in range(k) if m != l])]
                ytrain = y[np.concatenate([k_fold_indices[m] for m in range(k) if m != l])]
                Xval = X[k_fold_indices[l]]
                yval = y[k_fold_indices[l]]
                a, b = dualHinge(Xtrain, ytrain, lamb, kernel_func)
                yhat = dualClassify(Xval, a, b, Xtrain, ytrain, lamb, kernel_func)
                cv_acc[l, i, j] = np.mean(yhat == yval)
    
    # TODO: compute the average accuracies over k folds
    # TODO: identify the best lamb and kernel function
    # TODO: return a "len(lamb_list)-by-len(kernel_list)" accuracy variable,
    # the best lamb and the best kernel
    avg_acc = np.mean(cv_acc, axis=0)
    best_lamb, best_kernal = np.unravel_index(np.argmax(avg_acc, axis=None), avg_acc.shape)
    return avg_acc, lamb_list[best_lamb], kernel_list[best_kernal].__name__

def poly2Kernel(X1, X2):
    return polyKernel(X1, X2, 3)

def gauss1Kernel(X1, X2):
    return gaussKernel(X1, X2, 5)

# avg_acc, best_lamb, best_kernel = cvMnist("C:\\Users\\osasu\\Desktop\\COMP3105\\A2files", [0.001, 0.01], [linearKernel, poly2Kernel, gauss1Kernel])
# print("avg_acc", avg_acc, sep="\n")
# print("best_lamb", best_lamb)
# print("best_kernel", best_kernel)