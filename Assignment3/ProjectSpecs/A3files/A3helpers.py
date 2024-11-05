# COMP 3105 Assignment 3
# Carleton University
# NOTE: We provide some helper functions here. 
#       See the A3 instructions for more information.
import numpy as np
from matplotlib import pyplot as plt


def augmentX(X):
    n = X.shape[0]
    return np.concatenate((np.ones((n, 1)), X),  axis=1)


def unAugmentX(X):
    # Assume the first dimension has all ones
    return X[:, 1:]


def convertToOneHot(y, n_classes):
    # y should be a 1d array
    y = y.astype(int).flatten()
    Y = np.eye(n_classes)[y]
    return Y.astype(float)


def generateData(n, gen_model, rand_seed=None):

    d = 2
    shift = 1.8

    if rand_seed is not None:
        np.random.seed(rand_seed)

    X = []
    y = []
    m = n // 4
    class_label = 0
    for i in [-1, 1]:
        for j in [-1, 1]:
            if gen_model == 1:
                X.append(np.random.randn(m, d) + 
                         class_label * shift)
            elif gen_model == 2:
                X.append(np.random.randn(m, d) + 
                         shift * np.array([[i, j]]))
            else:
                raise ValueError("Unknown generative model")
            y.append(np.ones((m, 1)) * class_label)
            class_label += 1
    X = np.vstack(X)
    y = np.vstack(y)

    return X, convertToOneHot(y, 4)


def plotPoints(X, Y):
    # plot the data points
    k = Y.shape[1]
    markers = ['o', '+', 'd', 'x', '^', 'v', 's']
    colors = ['r', 'b', 'g', 'y', 'm', 'c', 'k']
    labels = Y.argmax(axis=1)
    for i in range(k):
        Xpart = X[labels == i]

        plt.scatter(Xpart[:, 0], Xpart[:, 1], 
                    marker=markers[i], 
                    color=colors[i],
                    label=f'class {i}')
    return


def getRange(X):
    x_min = np.amin(X[:, 0]) - 0.1
    x_max = np.amax(X[:, 0]) + 0.1
    y_min = np.amin(X[:, 1]) - 0.1
    y_max = np.amax(X[:, 1]) + 0.1
    return x_min, x_max, y_min, y_max


def plotModel(X, Y, W, classify):

    X = unAugmentX(X)
    plotPoints(X, Y)

    # plot model
    x_min, x_max, y_min, y_max = getRange(X)
    grid_step = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))
    Y = classify(np.c_[np.ones(len(xx.ravel())), xx.ravel(), yy.ravel()], W)
    labels = Y.argmax(axis=1)

    # Put the result into a color plot
    labels = labels.reshape(xx.shape)
    plt.contourf(xx, yy, labels, 
                 colors=['r', 'r', 'b', 'b', 'g', 'g', 'y', 'y'], 
                 alpha=0.3)
    plt.legend()
    plt.show()
    return


def synClsExperiments(minMulDev, classify, calculateAcc):
    n_runs = 10
    n_test = 1000
    n_train_list = [16, 32, 64, 128]
    gen_model_list = [1, 2]
    train_acc = np.zeros([len(n_train_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(n_train_list), len(gen_model_list), n_runs])
    for r in range(n_runs):
        for i, n_train in enumerate(n_train_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, Ytrain = generateData(n=n_train, gen_model=gen_model)
                Xtest, Ytest = generateData(n=n_test, gen_model=gen_model)
                
                Xtrain = augmentX(Xtrain)
                Xtest = augmentX(Xtest)

                W = minMulDev(Xtrain, Ytrain)
                Yhat = classify(Xtrain, W)
                train_acc[i, j, r] = calculateAcc(Yhat, Ytrain)
                Yhat = classify(Xtest, W)
                test_acc[i, j, r] = calculateAcc(Yhat, Ytest)

    train_acc = np.mean(train_acc, axis=2)
    test_acc = np.mean(test_acc, axis=2)
    
    return train_acc, test_acc


def plotImg(x):
    img = x.reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.show()
    return


def plotImgs(X):
    # plot the first 20 images of X
    fig = plt.figure(figsize=(16, 6))
    for i in range(20):
        ax = fig.add_subplot(3, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(X[i].reshape((28, 28)), cmap='gray')
    plt.show()
    return
