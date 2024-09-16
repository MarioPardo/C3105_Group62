# The functions in this cell are the answers to the assignment
import os

# import numpy as np
import autograd.numpy as np  # when testing gradient
from cvxopt import matrix, solvers
import pandas as pd


solvers.options['show_progress'] = False  # silence cvxopt

######## Q1 ########
# Q1a
def minimizeL2(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)

# Q1b
def minimizeL1(X, y):
    # first d variables are w, second n variables are delta
    n, d = X.shape

    c = np.concatenate([np.zeros(d), np.ones(n)]) # sum the deltas
    c = matrix(c)

    G = np.concatenate([np.concatenate([X, -np.eye(n)], axis=1),
                        np.concatenate([-X, -np.eye(n)], axis=1),
                        np.concatenate([np.zeros_like(X), -np.eye(n)], axis=1)])
    G = matrix(G)

    h = np.concatenate([y, -y, np.zeros_like(y)])
    h = matrix(h)

    res = solvers.lp(c, G, h)
    if res['status'] != 'optimal' and res['status'] != 'unknown':  # for our own debug purpose
        raise ValueError("Something's wrong with cvxopt solver")
    w1 = res['x'][:d]
    return np.array(w1)

# Q1c
def minimizeLinf(X, y):
    # first d variables are w, the last variable is delta
    n, d = X.shape
    c = np.concatenate([np.zeros(d), [1]])
    c = matrix(c)

    last_G = np.zeros(d + 1)
    last_G[d] = -1
    G = np.concatenate([np.concatenate([X, -np.ones((n, 1))], axis=1),
                        np.concatenate([-X, -np.ones((n, 1))], axis=1),
                        last_G[None, :]])
    G = matrix(G)

    h = np.concatenate([y, -y, np.zeros(1)[None, :]])
    h = matrix(h)

    res = solvers.lp(c, G, h)
    if res['status'] != 'optimal' and res['status'] != 'unknown':  # for our own debug purpose
        raise ValueError("Something's wrong with cvxopt solver")
    winf = res['x'][:d]
    return np.array(winf)


# Q1d
def loss_L2(w, X, y):
    return (np.linalg.norm(X @ w - y) ** 2) / (2 * X.shape[0])


def loss_L1(w, X, y):
    return np.mean(np.abs(X @ w - y))


def loss_Linf(w, X, y):
    return np.amax(np.abs(X @ w - y))


def compute_reg_losses(w_L2, w_L1, w_Linf, X, y):
    np_results = np.zeros([3, 3])

    np_results[0, 0] = loss_L2(w_L2, X, y)
    np_results[1, 0] = loss_L2(w_L1, X, y)
    np_results[2, 0] = loss_L2(w_Linf, X, y)

    np_results[0, 1] = loss_L1(w_L2, X, y)
    np_results[1, 1] = loss_L1(w_L1, X, y)
    np_results[2, 1] = loss_L1(w_Linf, X, y)

    np_results[0, 2] = loss_Linf(w_L2, X, y)
    np_results[1, 2] = loss_Linf(w_L1, X, y)
    np_results[2, 2] = loss_Linf(w_Linf, X, y)

    return np_results


def synRegExperiments():

    def genData(n_points):
        '''
        This function generate synthetic data
        '''
        X = np.random.randn(n_points, d)  # input matrix
        X = np.concatenate((np.ones((n_points, 1)), X),  axis=1)  # augment input
        y = X @ w_true + np.random.randn(n_points, 1) * noise  # ground truth label
        return X, y

    n_runs = 100
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2
    train_loss = np.zeros([n_runs, 3, 3])  # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3])  # n_runs * n_models * n_metrics
    for r in range(n_runs):

        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train)
        Xtest, ytest = genData(n_test)

        # Learn different models from the training data
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        # TODO: Evaluate the three models' performance (for each model, 
        #       calculate the L2, L1 and L infinity losses on the training 
        #       data). Save them to `train_loss`
        train_loss[r] = compute_reg_losses(w_L2, w_L1, w_Linf, Xtrain, ytrain)

        # TODO: Evaluate the three models' performance (for each model, 
        #       calculate the L2, L1 and L infinity losses on the test 
        #       data). Save them to `test_loss`
        test_loss[r] = compute_reg_losses(w_L2, w_L1, w_Linf, Xtest, ytest)

    # TODO: compute the average losses over runs
    # TODO: return a 3-by-3 training loss variable and a 3-by-3 test loss variable
    return np.mean(train_loss, axis=0), np.mean(test_loss, axis=0)


######## Q2 ########
# Q2a
def linearRegL2Obj(w, X, y):

    yhat = X @ w
    delta = yhat - y
    obj_val = 0.5 * np.mean(delta ** 2)
    grad = X.T @ delta / X.shape[0]
    return obj_val, grad


# Q2b
def gd(func, w_init, X, y, step_size, max_iter, tol=1e-10):

    w = w_init

    for _ in range(max_iter):
        obj_val, grad = func(w, X, y)
        if np.linalg.norm(grad) < tol:
            break
        w = w - step_size * grad

    return w


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


# Q2c
def logisticRegObj(w, X, y):
    # X: n x d
    # y: n x 1 (in {0, 1})
    # Note for over/underflow
    zhat = X @ w
    yhat = sigmoid(zhat)

    obj_val = np.mean(y * np.logaddexp(0, -zhat)
                      + (1 - y) * np.logaddexp(0, zhat))
    grad = (X.T @ (yhat - y) ) / X.shape[0]
    return obj_val, grad


# Q2d
def synClsExperiments():

    def genData(n_points, d):
        '''
        This function generate synthetic data
        '''
        c0 = np.ones([1, d])  # class 0 center
        c1 = -np.ones([1, d])  # class 1 center
        X0 = np.random.randn(n_points, d) + c0  # class 0 input
        X1 = np.random.randn(n_points, d) + c1  # class 1 input
        X = np.concatenate((X0, X1), axis=0)
        X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1)  # augmentation
        y = np.concatenate([np.zeros([n_points, 1]), np.ones([n_points, 1])], axis=0)
        return X, y

    def runClsExp(m=100, d=2, eta=0.1, max_iter=1000, tol=1e-10):
        '''
        Run classification experiment with the specified arguments
        '''

        Xtrain, ytrain = genData(m, d)
        n_test = 1000
        Xtest, ytest = genData(n_test, d)

        w_init = np.random.randn(d + 1, 1)
        w_logit = gd(logisticRegObj, w_init, Xtrain, ytrain, eta, max_iter, tol)
        ytrain_hat = 0.5 * (1 + np.sign(Xtrain @ w_logit))
        train_acc = np.sum(ytrain == ytrain_hat) / (2 * m)

        ytest_hat = 0.5 * (1 + np.sign(Xtest @ w_logit))
        test_acc = np.sum(ytest == ytest_hat) / (2 * n_test)

        return train_acc, test_acc

    n_runs = 100
    train_acc = np.zeros([n_runs, 4, 3])
    test_acc = np.zeros([n_runs, 4, 3])
    for r in range(n_runs):
        for i, m in enumerate((10, 50, 100, 200)):
            train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
        for i, d in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(d=d)
        for i, eta in enumerate((0.1, 1.0, 10., 100.)):
            train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(eta=eta)

    return np.mean(train_acc, axis=0), np.mean(test_acc, axis=0)


######## Q3 ########
# Q3a
def preprocessAutoMPG(dataset_folder):

    df_data = pd.read_csv(os.path.join(dataset_folder, "auto-mpg.data"), 
                          header=None, 
                          delim_whitespace=True)
    # Drop `origin`` and `car name` features
    del df_data[8]
    del df_data[7]
    # Drop points with missing values: 32, 126, 330, 336, 354, 374
    df_data = df_data.drop([32, 126, 330, 336, 354, 374])

    # Convert labels and then remove them from inputs
    y = df_data[0].to_numpy(float)[:, None]
    del df_data[0]
    X = df_data.to_numpy(float)

    return X, y


# Q3b
def runAutoMPG(dataset_folder):

    X, y = preprocessAutoMPG(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X),  axis=1)  # augment

    n_runs = 100
    train_loss = np.zeros([n_runs, 3, 3])  # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 3, 3])  # n_runs * n_models * n_metrics

    for r in range(n_runs):

        # TODO: Randomly partition the dataset into two parts (50% 
        #       training and 50% test)
        n_train = round(n * 0.5)
        indices = np.random.permutation(n)
        training_idx, test_idx = indices[:n_train], indices[n_train:]
        Xtrain, Xtest = X[training_idx,:], X[test_idx,:]
        ytrain, ytest = y[training_idx,:], y[test_idx,:]
    
        # TODO: Learn three different models from the training data
        #       using L1, L2 and L infinity losses
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_L1 = minimizeL1(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)
    
        # TODO: Evaluate the three models' performance (for each model, 
        #       calculate the L2, L1 and L infinity losses on the training 
        #       data). Save them to `train_loss`
        train_loss[r] = compute_reg_losses(w_L2, w_L1, w_Linf, Xtrain, ytrain)

        # TODO: Evaluate the three models' performance (for each model, 
        #       calculate the L2, L1 and L infinity losses on the test 
        #       data). Save them to `test_loss`
        test_loss[r] = compute_reg_losses(w_L2, w_L1, w_Linf, Xtest, ytest)

    # TODO: compute the average losses over runs
    # TODO: return a 3-by-3 training loss variable and a 3-by-3 test loss variable
    return np.mean(train_loss, axis=0), np.mean(test_loss, axis=0)


# Q3c
def preprocessSonar(dataset_folder):

    df_data = pd.read_csv(os.path.join(dataset_folder, "sonar.all-data"), 
                          header=None)
    # Convert labels
    df_data[60] = (df_data[60] == 'R').astype(float)
    y = df_data[60].to_numpy()[:, None]
    # Then remove labels from the data
    del df_data[60]
    X = df_data.to_numpy()

    return X, y


# Q3d
def runSonar(dataset_folder):

    X, y = preprocessSonar(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X),  axis=1)  # augment

    eta_list = [0.1, 1, 10, 100]
    train_acc = np.zeros([len(eta_list)])
    val_acc = np.zeros([len(eta_list)])
    test_acc = np.zeros([len(eta_list)])

    # TODO: Randomly partition the dataset into three parts (40%
    #       training (use the round function), 40% validation and 
    #       the remaining ~20% as test)
    n_train = round(n * 0.4)
    n_val = round(n * 0.4)
    indices = np.random.permutation(n)
    training_idx, val_idx, test_idx = indices[:n_train], indices[n_train:(n_train+n_val)], indices[(n_train+n_val):]
    Xtrain, Xval, Xtest = X[training_idx,:], X[val_idx,:], X[test_idx,:]
    ytrain, yval, ytest = y[training_idx,:], y[val_idx,:], y[test_idx,:]

    for i, eta in enumerate(eta_list):

        w_init = np.zeros([d + 1, 1])
        w = gd(logisticRegObj, w_init, Xtrain, ytrain, eta, max_iter=1000, tol=1e-8)

        # TODO: Evaluate the model's accuracy on the training
        #       data. Save it to `train_acc`
        y_hat = 0.5 * (1 + np.sign(Xtrain @ w))
        train_acc[i] = np.sum(ytrain == y_hat) / Xtrain.shape[0]

        # TODO: Evaluate the model's accuracy on the validation
        #       data. Save it to `val_acc`
        y_hat = 0.5 * (1 + np.sign(Xval @ w))
        val_acc[i] = np.sum(yval == y_hat) / Xval.shape[0]

        # TODO: Evaluate the model's accuracy on the test
        #       data. Save it to `test_acc`
        y_hat = 0.5 * (1 + np.sign(Xtest @ w))
        test_acc[i] = np.sum(ytest == y_hat) / Xtest.shape[0]

    return train_acc, val_acc, test_acc
