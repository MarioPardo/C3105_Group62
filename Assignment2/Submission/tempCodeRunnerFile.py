
y_pred_mario = classify(X, q1a, q1a0)
y_pred_dante = classify(X, q1aDante, q1a0Dante)

#q1d
def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)
