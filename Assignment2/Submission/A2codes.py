
#COMP3105 A2 
#

##Prereq Stuff--------------
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt


def linearKernel(X1, X2):
    return X1 @ X2.T


def polyKernel(X1, X2, degree):
    return (X1 @ X2.T + 1) ** degree


def gaussKernel(X1, X2, width):
    distances = cdist(X1, X2, 'sqeuclidean')
    return np.exp(- distances / (2*(width**2)))


def generateData(n, gen_model):

    # Controlling the random seed will give you the same 
    # random numbers every time you generate the data. 
    # The seed controls the internal random number generator (RNG).
    # Different seeds produce different random numbers. 
    # This can be handy if you want reproducible results for debugging.
    # For example, if your code *sometimes* gives you an error, try
    # to find a seed number (0 or others) that produces the error. Then you can
    # debug your code step-by-step because every time you get the same data.

    # np.random.seed(0)  # control randomness when debugging

    if gen_model == 1 or gen_model == 2:
        # Gen 1 & 2
        d = 2
        w_true = np.ones([d, 1])

        X = np.random.randn(n, d)

        if gen_model == 1:
            y = np.sign(X @ w_true)  # generative model 1
        else:
            y = np.sign((X ** 2) @ w_true - 1)  # generative model 2

    elif gen_model == 3:
        # Gen 3
        X, y = generateMoons(n)

    else:
        raise ValueError("Unknown generative model")

    return X, y


def generateMoons(n, noise=0.1):
    n_samples_out = n // 2
    n_samples_in = n - n_samples_out
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [np.append(outer_circ_x, inner_circ_x), 
         np.append(outer_circ_y, inner_circ_y)]
    ).T
    X += np.random.randn(*X.shape) * noise

    y = np.hstack(
        [-np.ones(n_samples_out, dtype=np.intp), 
         np.ones(n_samples_in, dtype=np.intp)]
    )[:, None]
    return X, y


def plotPoints(X, y):
    # plot the data points from two classes
    X0 = X[y.flatten() >= 0]
    X1 = X[y.flatten() < 0]

    plt.scatter(X0[:, 0], X0[:, 1], marker='x', label='class -1')
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', label='class +1')
    return


def getRange(X):
    x_min = np.amin(X[:, 0]) - 0.1
    x_max = np.amax(X[:, 0]) + 0.1
    y_min = np.amin(X[:, 1]) - 0.1
    y_max = np.amax(X[:, 1]) + 0.1
    return x_min, x_max, y_min, y_max


def plotModel(X, y, w, w0, classify):

    plotPoints(X, y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = classify(np.c_[xx.ravel(), yy.ravel()], w, w0)

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.legend()
    plt.show()
    return


def plotAdjModel(X, y, a, a0, kernel_func, adjClassify):

    plotPoints(X, y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = adjClassify(np.c_[xx.ravel(), yy.ravel()], a, a0, X, kernel_func)

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.legend()
    plt.show()
    return


def plotDualModel(X, y, a, b, lamb, kernel_func, dualClassify):

    plotPoints(X, y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    z = dualClassify(np.c_[xx.ravel(), yy.ravel()], a, b, X, y, 
                     lamb, kernel_func)

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.RdBu, alpha=0.5)
    plt.legend()
    plt.show()

    return


def plotDigit(x):
    img = x.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
    return


#-----------

#-----Q1-----
# a)

import numpy as np
from scipy.optimize import minimize
from cvxopt import matrix, solvers

def obj_loss_function(params,X,y,lamb):

    w = params[:-1]  
    w0 = params[-1]  

    logits = X @ w + w0  # Xw + w0
    exponent = -y * logits  
    log_loss = np.sum(np.log(1 + np.exp(exponent))) 

    reg_term = (lamb / 2) * np.sum(w**2)
    
    total_loss = log_loss + reg_term
    return total_loss



def minBinDev(X, y, lamb):
    #want to return w, w0
    n, d = X.shape
    
    initial_params = np.zeros(d + 1)  

    results = minimize(obj_loss_function,initial_params,args = (X,y,lamb), method = 'bfgs')

    w_opt = results.x[:-1]  #  d elements are w
    w0_opt = results.x[-1]  # last element is w0
    
    return w_opt, w0_opt



#b)


def minHinge(X, y, lamb, stabilizer=1e-5):
    n,d = X.shape

    y = y.flatten()

    #P in (P,q,G,h)
    P = np.zeros((n+d+1,n+d+1))

    #for size of dimensioanlity, create that size identity matrix in the top left
    P[:d, :d] = lamb * np.eye(d)

    #q = d+1 zeroes, n 1's
    q = np.hstack([np.zeros(d+1), np.ones(n)])

    #G matrix: 2n rows for the slack variables for 2 constraints
    G = np.zeros((2*n, d+1+n))

    #first constraint, G11 (could be 0)
    G[:n, :d] = -y[:, None] * X

    #g12 for w0 (could also be 0)
    G[:n,d]= -y

    #g13, Iden matrix (prob good)
    G[:n,d+1:]= -np.eye(n) #mario agrees

    #second contraint G23
    G[n:, d+1:] = -np.eye(n) #mario agrees

    #h, constraint for G, so 2n rows to match, 0n and -1n.
    h = np.hstack([-np.ones(n), np.zeros(n)]) 

    P += stabilizer * np.eye(d+1+n)

    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    solution = solvers.qp(P,q,G,h)

    w = np.array(solution['x'][:d])
    w0 = np.array(solution['x'][d])

    return w,w0


#c

def classify(Xtest, w, w0):

    matrix_weight = np.dot(Xtest,w)+w0

    yhat = np.sign(matrix_weight)

    return yhat


#d









#-----Q2-----------------------
#
#



#a

def objective_function(params, y, lamb, K):
    n = len(y)
    
    alpha = params[:n]
    alpha = alpha[:,None]
    alpha0 = params[n]
    
    linear_combination = (K @ alpha) + alpha0
    loss = np.sum(np.logaddexp(0, -y * linear_combination))
    
    regularization = (lamb / 2) * float(alpha.T @ K @ alpha)
    return loss + regularization


def adjBinDev(X, y, lamb, kernel_func):
    n, d = X.shape
    K = kernel_func(X, X)
    initial_params = np.ones(n + 1)
    
    result = minimize(objective_function, initial_params, args=(y, lamb, K))
    
    a = result.x[:-1][:,None]
    a0 = result.x[-1]

    return a, a0

#b

def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n, d = X.shape
    #print("n = ", n)
    #print("d = ", d)
    K = kernel_func(X, X) 
    


    P = np.zeros((2*n + 1, 2*n+1)) 
    P[:n, :n] = lamb * K  # Kernel matrix for alpha terms
    P = matrix(P + stabilizer * np.eye(2*n+1))  # Stabilization

    q = matrix(np.hstack([np.zeros(n + 1),  np.ones(n)]))
    
    # Create G matrix
    # G1: For the non-negativity constraints of the slack variables 
    G11 = np.zeros([n,n]) #for 1, its nxd
    G12 = np.zeros([n,1])
    print("G12 shape", G12.shape)
    G13 = -np.eye(n)
    print("G13 shape", G13.shape)
    G1 = np.hstack([G11,G12,G13])
    print("G1 shape", G1.shape)
    
    # G2: For the hinge constraints
    G21 = -y * K
    G22 = -y * np.ones([n,1])
    print("G22 shape", G22.shape)
    G23 = -np.eye(n)
    print("G23 shape", G23.shape)
    G2 = np.hstack([G21,G22,G23])
    print("G2 shape", G2.shape)

    G = np.vstack([G1, G2])  # Stack G1 and G2 to form the full G matrix

    # Create the h vector
    h = np.concatenate([np.zeros(n), -np.ones(n)]) 

    # Convert 
    G = matrix(G)
    h = matrix(h)

    solution = solvers.qp(P, q, G, h)

    # Extract solutions for α and α_0
    alphas = np.array(solution['x'][:n])
    alpha0 = np.array(solution['x'][n])

    return alphas, alpha0

#q2c
def adjClassify(Xtest, a, a0, X, kernel_func):
    return np.sign( (kernel_func(Xtest,X)@ a + a0)  )

#q2d


def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def predict(X, alphas, alpha0, kernel, X_support):
    K = kernel(X_support, X)
    predictions = K.T.dot(alphas) + alpha0
    return np.sign(predictions)

def sunExperimentsKernel():
    n_runs = 10 
    n_train = 10
    n_test = 100
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
                
                # BinDev model
                a, a0 = adjBinDev(Xtrain, ytrain, lamb, kernel)
                train_pred_bindev = predict(Xtrain, a, a0, kernel, Xtrain)
                test_pred_bindev = predict(Xtest, a, a0, kernel, Xtrain)
                train_acc_bindev[i, j, r] = compute_accuracy(ytrain, train_pred_bindev)
                test_acc_bindev[i, j, r] = compute_accuracy(ytest, test_pred_bindev)
                
                # Hinge model
                
                a, a0 = adjHinge(Xtrain, ytrain, lamb, kernel)
                train_pred_hinge = predict(Xtrain, a, a0, kernel, Xtrain)
                test_pred_hinge = predict(Xtest, a, a0, kernel, Xtrain)
                train_acc_hinge[i, j, r] = compute_accuracy(ytrain, train_pred_hinge)
                test_acc_hinge[i, j, r] = compute_accuracy(ytest, test_pred_hinge)
                

    # Compute average accuracies over runs
    train_acc = np.hstack([np.mean(train_acc_bindev,axis = 2), np.mean(train_acc_hinge,axis=2)])
    test_acc = np.hstack([np.mean(test_acc_bindev,axis = 2), np.mean(test_acc_hinge,axis=2)])
    
    # Return or print results
    return train_acc, test_acc


















##Testing 1b, 2b
solvers.options['show_progress'] = False


X,y = generateData(100,2)
lamb = 1.0  # Regularization parameter

#linear kernel
kernel_func = lambda X1, X2: np.dot(X1, X2.T)

#q1a, q1a0 = MarioMinHinge(X,y,lamb)
q1aDante, q1a0Dante = minHinge(X,y,lamb)
q2a, q2a0 = adjHinge(X, y, lamb,kernel_func)

print("Optimized weights Q1a Dante:", q1aDante.flatten())
#print("Opt Weights q1 mario", q1aDante.flatten())
print("Optimized weights Q2:", q2a.flatten())

print("Optimized bias Q1:", q1a0Dante)
print("Optimized bias Q2:", q2a0)

#y_pred_mario = classify(X, q1a, q1a0)
y_pred_dante = classify(X, q1aDante, q1a0Dante)

#q1d
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

#print("MarioMinHinge Accuracy:", compute_accuracy(y, y_pred_mario))

print("minHinge Accuracy:", compute_accuracy(y, y_pred_dante))