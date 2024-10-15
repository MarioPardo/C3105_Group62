
#COMP3105 A2 
#
#

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

#Mario Version
def minHinge(X, y, lamb, stabilizer=1e-5):
    n, d = X.shape

    # Define P matrix for the quadratic term
    P = np.zeros((d + 1 + n, d + 1 + n))
    P[:d, :d] = lamb * np.eye(d)  # Regularization on the weights

    # Define q vector for the linear term
    q = np.hstack([np.zeros(d + 1), np.ones(n)])

    # Define G matrix for the constraints
    G = np.zeros((2 * n, d + 1 + n))
    G[:n, d+1:] = -np.eye(n)  # Constraints for non-negativity of slack variables
    G[n:, :d] = np.diag(y) @ X
    G[n:, d] = y
    G[n:, d+1:] = -np.eye(n)  # Constraints for hinge loss

    # Define h vector for the constraints
    h = np.hstack([np.zeros(n), -np.ones(n)])

    # Add a stabilizer to ensure P is positive definite
    P += stabilizer * np.eye(d + 1 + n)

    # Convert all to CVXOPT matrices
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)

    # Suppress solver progress messages
    solvers.options['show_progress'] = False

    # Solve the quadratic programming problem
    solution = solvers.qp(P, q, G, h)

    # Extract weights and bias from the solution
    w = np.array(solution['x'][:d])
    w0 = np.array(solution['x'][d])

    return w, w0

#c

def classify(Xtest, w, w0):

    matrix_weight = np.dot(Xtest,w)+w0

    yhat = np.sign(matrix_weight)

    return yhat


#d







#-----Q2



#a
def objective_function(params, y, lamb, K):
    n = len(y)
    
    alpha = params[:n]
    alpha0 = params[n]
    
    linear_combination = (K @ alpha) + alpha0
    loss = np.sum(np.logaddexp(0, -y * linear_combination))
    
    regularization = (lamb / 2) * alpha.T @ K @ alpha
    return loss + regularization


def adjBinDev(X, y, lamb, kernel_func):
    n, d = X.shape
    K = kernel_func(X, X)
    initial_params = np.ones(n + 1)
    
    result = minimize(objective_function, initial_params, args=(y, lamb, K))
    
    a = result.x[:-1]
    a0 = result.x[-1]

    return a, a0




#b

def adjHinge(X, y, lamb, kernel_func, stabilizer=1e-5):
    n, d = X.shape
    #print("n = ", n)
    #print("d = ", d)
    K = kernel_func(X, X) 

    y = np.array(y, dtype=np.double)
    #print("Shape of y : ", y.shape)


    P = np.zeros((2*n + 1, 2*n+1)) 
    P[:n, :n] = lamb * K  # Kernel matrix for alpha terms
    P = matrix(P + stabilizer * np.eye(2*n+1))  # Stabilization

    q = matrix(np.hstack([np.zeros(n + 1),  np.ones(n)]))
    
    # Create G matrix
    # G1: For the non-negativity constraints of the slack variables 
    G11 = np.zeros([n,n]) #for 1, its nxd
    G12 = np.zeros([n,1])
    G13 = -np.eye(n)
    G1 = np.hstack([G11,G12,G13])
    
    # G2: For the hinge constraints
    G21 = -y * K
    G22 = -y * np.ones([n,1])
    G23 = -np.eye(n)
    G2 = np.hstack([G21,G22,G23])

    G = np.vstack([G1, G2])  # Stack G1 and G2 to form the full G matrix

    # Create the h vector
    h = np.concatenate([np.zeros(n), -np.ones(n)]) 

    # Convert 
    P = matrix(P)
    q = matrix(q)
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














##Testing 1b, 2b
solvers.options['show_progress'] = False


X = np.random.randn(100, 2)  # 100 samples, 2 features each
y = np.random.choice([-1, 1], size=100)  # Binary labels
lamb = 1.0  # Regularization parameter

#linear kernel
kernel_func = lambda X1, X2: np.dot(X1, X2.T)

q1a, q1a0 = minHinge(X,y,lamb)
q2a, q2a0 = adjHinge(X, y, lamb,kernel_func)

print("Optimized weights Q1:", q2a.flatten())
print("Optimized weights Q2:", q2a.flatten())

print("Optimized bias Q1:", q1a0)
print("Optimized bias Q2:", q2a0)