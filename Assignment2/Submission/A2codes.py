
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
def MarioMinHinge(X, y, lamb, stabilizer=1e-5):
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


#q1b

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

    #second contraint 
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







#-----Q2



#a
def objective_function(params, y, lamb, K):
    n = len(y)
    
    alpha = params[:n]
    alpha0 = params[n]
    
    linear_combination = K @ alpha + alpha0
    loss = np.log(1 + np.exp(-y * linear_combination)).sum()
    
    regularization = (lamb / 2) * np.dot(alpha.T, K @ alpha)
    return loss + regularization


def adjBinDev(X, y, lamb, kernel_func):
    n, d = X.shape
    K = kernel_func(X, X)
    initial_params = np.zeros(n + 1)
    
    result = minimize(objective_function, initial_params, args=(y, lamb, K))
    
    alpha = result.x[:-1]
    alpha0 = result.x[-1]

    return alpha, alpha0




#b

def adjHinge(X,y,lamb,kernel_func,stabilizer=1e-5):
    n = len(y)
    K = kernel_func(X,X)  # Compute the kernel matrix
    y = np.array(y, dtype=np.double)

    # Create the P matrix
    P = np.zeros((n + 1 + n, n + 1 + n))  # α, α_0, ξ
    P[:n, :n] = K  # Kernel matrix for α terms
    #P = P * np.outer(y, y)  # Modulate by outer product of labels
    P += stabilizer * np.eye(n + 1 + n)  # Stabilization

    #1 vector
    q = np.hstack([np.zeros(n + 1), lamb * np.ones(n)])
    
    #G matrix
    G = np.zeros((2 * n, n + 1 + n))
    
    G[:n, n+1:] = np.eye(n)  # For ξ ≥ 0
    G[n:, :n] = -K * y[:, None]  # -Δ(y)K part of the hinge constraint
    G[n:, n] = -y  # -Δ(y)α_0 part
    G[n:, n+1:] = np.eye(n)  #slack variables
    
    # Create the h vector
    h = np.hstack([np.zeros(n), -np.ones(n)])  # 0s for ξ ≥ 0 and 1s for the hinge constraint

    # Convert 
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
  

    #Solve
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h)

    # Extract solutions
    alphas = np.array(solution['x'][:n])
    alpha0 = np.array(solution['x'][n])

    return alphas, alpha0




##Testing 1b, 2b
solvers.options['show_progress'] = False


X = np.random.randn(100, 2)  # 100 samples, 2 features each
y = np.random.choice([-1, 1], size=100)  # Binary labels
lamb = 1.0  # Regularization parameter

#linear kernel
kernel_func = lambda X1, X2: np.dot(X1, X2.T)

q1a, q1a0 = MarioMinHinge(X,y,lamb)
q1aDante, q1a0Dante = minHinge(X,y,lamb)
q2a, q2a0 = adjHinge(X, y, lamb,kernel_func)

print("Optimized weights Q1:", q1a.flatten(), q1aDante.flatten())
print("Optimized weights Q2:", q2a.flatten())

print("Optimized bias Q1:", q1a0, q1a0Dante)
print("Optimized bias Q2:", q2a0)

y_pred_mario = classify(X, q1a, q1a0)
y_pred_dante = classify(X, q1aDante, q1a0Dante)

#q1d
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

print("MarioMinHinge Accuracy:", compute_accuracy(y, y_pred_mario))
print("minHinge Accuracy:", compute_accuracy(y, y_pred_dante))