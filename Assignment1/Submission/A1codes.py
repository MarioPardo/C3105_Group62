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

#d

def l2_loss(w, X,y):

    predicted_diff = (np.linalg.norm(X @ w - y)**2)
    relative_result = predicted_diff / (2*X.shape[0])
    return relative_result


def l1_loss(w, X,y):
    predicted = np.abs(X@w - y)
    avg = np.mean(predicted)
    return avg

def linf_loss(w, X,y):
    diff = np.abs(X@w-y)
    max_val = np.amax(diff)
    return max_val


def compute_loss(w_l2, w_l1,w_Linf,X,y):
    results = np.zeros([3, 3])

    results[0,0] = l2_loss(w_l2, X,y)
    results[1,0] = l2_loss(w_l1,X,y)
    results[2,0] = l2_loss(w_Linf, X, y)
    
    results[0,1] = l1_loss(w_l2, X,y)
    results[1,1] = l1_loss(w_l1,X,y)
    results[2,1] = l1_loss(w_Linf, X, y)

    results[0,2] = linf_loss(w_l2, X,y)
    results[1,2] = linf_loss(w_l1,X,y)
    results[2,2] = linf_loss(w_Linf, X, y)

    return results



def synRegExperiments():
    def genData(n_points): 
        X = np.random.randn(n_points, d) # input matrix
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
        y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label 
        return X, y
    
    n_runs = 100 
    n_train = 30 
    n_test = 1000
    d=5
    noise = 0.2
    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics 
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics

    for r in range(n_runs):
        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train)
        Xtest, ytest = genData(n_test)
        # Learn different models from the training data
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        
        train_loss[r] = compute_loss(w_L2, w_L1,w_Linf,Xtrain,ytrain)

        
        test_loss[r] = compute_loss(w_L2, w_L1,w_L1,Xtest,ytest)

       # Compute the average losses over runs
    avg_train_loss = np.mean(train_loss, axis=0)
    avg_test_loss = np.mean(test_loss, axis=0)

    # Print the results
    print("Average Training Loss (3x3 Matrix):")
    print(avg_train_loss)
    print("\nAverage Test Loss (3x3 Matrix):")
    print(avg_test_loss)

    # Return the average losses
    return avg_train_loss, avg_test_loss

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
            break

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






##Q3


#Q3a

def preprocessAutoMPG(dataset_folder):

    data_df = pd.read_csv(os.path.join(dataset_folder, "auto-mpg.data"), 
                          header=None, 
                          delim_whitespace=True)
    
    print(data_df.head())


    del data_df[8]    #car name
    del data_df[7]  #origin

    df_data = data_df.dropna() #drop rows with any missing data

    labels = df_data[0].to_numpy(float)[:, None]
    del df_data[0]

    matrix = df_data.to_numpy(float)

    return matrix, labels



#Q3b

def runAutoMPG(dataset_folder):

    X, y = preprocessAutoMPG(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
    n_runs = 100
    train_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3]) # n_runs * n_models * n_metrics
    
    for r in range(n_runs):
        
        #partition data
        rand_indices = np.random.permutation(n)
        num_training = n // 2
        train_indices = rand_indices[:num_training]
        test_indices = rand_indices[num_training:]

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        l1model = l1_loss(X_train,y_train)
        l2model = l2_loss(X_train,y_train)
        linfmodel = linf_loss(X_train,y_train)

        train_loss = compute_loss(l2model,l1model,linfmodel,X_train,y_train)
        test_loss = compute_loss(l2model,l1model,linfmodel,X_test,y_test)

        avg_train_losses = np.mean(train_loss, axis=0)
        avg_test_losses = np.mean(test_loss, axis=0)


        return avg_train_losses, avg_test_losses


#Q3c

def preprocessSonar(dataset_folder):

    data_df = pd.read_csv(os.path.join(dataset_folder, "sonar.all-data"), 
                          header=None, 
                          delim_whitespace=True)


    # Convert labels
    data_df[60] = (data_df[60] == 'R').astype(float)
    y = data_df[60].to_numpy()[:, None]
    
    #remove labels from the data
    del data_df[60]
    X = data_df.to_numpy()

    return X, y



#Q3d
def runSonar(dataset_folder):

    X, y = preprocessSonar(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment
    eta_list = [0.1, 1, 10, 100]

    train_acc = np.zeros([len(eta_list)])
    val_acc = np.zeros([len(eta_list)])
    test_acc = np.zeros([len(eta_list)])

    indices = np.random.permutation(n)
    train_size = round(n * 0.4)
    val_size = round(n * 0.4)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    Xtrain, ytrain = X[train_indices], y[train_indices]
    Xval, yval = X[val_indices], y[val_indices]
    Xtest, ytest = X[test_indices], y[test_indices]


    for i, eta in enumerate(eta_list):
        w_init = np.zeros([d + 1, 1])
        w = gd(logisticRegObj, w_init, Xtrain, ytrain, eta, max_iter=1000, tol=1e-8)

        # TODO: Evaluate the model's accuracy on the training
        # data. Save it to `train_acc`
        train_pred = (1/2) * (1 * np.sign(Xtrain @ w))
        train_acc[i] = np.sum(ytrain == train_pred) / Xtrain.shape[0]

        # TODO: Evaluate the model's accuracy on the validation
            #       data. Save it to `val_acc`
        val_pred = 0.5 * (1 + np.sign(Xval @ w))
        val_acc[i] = np.sum(yval == val_pred) / Xval.shape[0]

            
        test_pred = 0.5 * (1 + np.sign(Xtest @ w))
        test_acc[i] = np.sum(ytest == test_pred) / Xtest.shape[0]

    return train_acc, val_acc, test_acc
