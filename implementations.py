import numpy as np
from numpy import linalg

# These are the functions to load/clean the data

def load_data(path):
    """Load data and convert it to the metrics system."""
    path_dataset = path
    raw_features = np.genfromtxt(path_dataset, delimiter=",", skip_header=1,usecols=[i for i in range(2,32)])
    raw_labels = np.genfromtxt(path_dataset, delimiter=",", skip_header=1,usecols=[1], dtype= str)
    Ids = np.genfromtxt(path_dataset, delimiter=",", skip_header=1,usecols=0, dtype=int)
    features = []
    n_features = raw_features.shape[1] 
    
    for i in range(0, n_features):
        features.append(raw_features[:, i])       
    features = np.asarray(features).T
    labels = np.asarray(list(map(lambda x: 0 if x == "b" else 1,raw_labels))).T
    
    return features, labels, Ids

def loadTrainData():
    return load_data("data/train.csv")
    
def loadTestData():
    return load_data("data/test.csv")

# Replace the -999.0 meaningless values by the median of the feature
def cleanData(data, medians): 
    cleaned = np.empty((data.shape[0], data.shape[1]))
    for i, line in enumerate(data):
        for j, elem in enumerate(line):
            cleaned[i][j] = elem if elem != -999.0 else medians[j]
    return cleaned

def mediansByFeature(data) :
    medians = [0] * data.shape[1]
    for idx, column in enumerate(data.T):
        good_values = []
        for elem in column:
            if(elem != -999.0):
                good_values.append(elem)
        medians[idx] = np.median(np.asarray(good_values))
    return medians


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x

def standardize_submission_dataset(x, mean, std) :
    return (x - mean)/std

def predictLabels(x_submission, submission_ids, w):
    predicted = x_submission.dot(w)
    y_submission = [0] * x_submission.shape[0]
    for i, elem in enumerate(predicted):
        if(elem >= 0.5):
            y_submission[i] = 1
        else:
            y_submission[i] = -1
            
    return zip(submission_ids, y_submission)

def standardize_submission_dataset(x, mean, std) :
    return (x - mean)/std


# Here are the 6 functions to implement:

#Note that for this one and least_squares_SGD, we actually implemented the gradient descent and 
#stochastic gradient descent algorithm
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Perform a gradient descent to find weights that minimize loss
    return the weights and associated loss"""
    loss = -1
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad = calculate_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        # update of w
        w = w - gamma * grad
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma, batch_size=1):
    """Perform a stochastic gradient descent to find weights that minimize loss
    return the weights and associated loss"""

    loss = -1
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_mse(y, tx, w)

    return w, loss

def least_squares(y, tx):
	"""calculate the least squares solution""" 
	a = tx.T.dot(tx)
	b = tx.T.dot(y)
	return np.linalg.solve(a, b)

def ridge_regression(y, tx, lambda_):
    """return the weights and associated loss"""
    aI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """return the weights and associated loss """
    threshold = 1e-8
    losses = []   
    tx = np.c_[np.ones((y.shape[0], 1)), tx]
    w = initial_w
    
    for i in range(max_iters):
        hess = calculate_hessian(y, tx, w)
        grad = calculate_gradient(y, tx, w)
        w -= gamma * np.linalg.solve(hess, grad)
        loss = calculate_loss(y, tx, w)
        losses.append(loss)
        #Check this condition before the next one, to avoid out of bounds access
        if(len(losses) > 1):
            #If the difference is loss is to small, we stop the algorithm and return weights and loss
            if (np.abs(losses[-1] - losses[-2]) < threshold):
                return w, loss

    return w, loss

def reg_logistic_regression(y, x, lambda_, initial_w, max_iters, gamma):
    """return the weights and loss Regularized logistics"""

    threshold = 1e-8
    losses = []
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
            loss, gradient = penalized_logistic_regression(y, tx, w, lambda_)
            losses.append(loss)
            w -= gamma * gradient   

            #Check this condition before the next one, to avoid out of bounds access
            if(len(losses) > 1):
                #If the difference is loss is to small, we stop the algorithm and return weights and loss
                if (np.abs(losses[-1] - losses[-2]) < threshold):
                    return w, loss
            
    return w, loss

#------These are some helpers methods for the 6 methods to implement------


def compute_mse(y, tx, w):
    """compute the loss by mse."""
    e = y - tx.dot(w)
    mse = 1/2*np.mean(e**2)
    return mse


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    epsilon = 1e-5 # we add a small epsilon in order to avoid an undefined '0' calculation in the log
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred+epsilon)) + (1 - y).T.dot(np.log(1 - pred+ epsilon))
    return np.squeeze(- loss)

def sigmoid(t):
    """apply sigmoid function on t."""
    sig = 0.5 * (1 + np.tanh(0.5 * t))
    return sig

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad


def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    r = np.multiply(pred, (1-pred))
    return tx.T.dot(r).dot(tx)


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss and gradient."""
    loss = calculate_loss(y, tx, w) + lambda_ * np.squeeze(w.T.dot(w))
    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, gradient


#------These are the functions to create the model--------------------------

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_ridge_regression(y, x, k_indices, k, lambda_):
    """k-1 groups are for the training, and we test on the last group
    return the loss and weights"""
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    w, _ = ridge_regression(y_tr, x_tr, lambda_)
    loss_te = compute_mse(y_te, x_te, w)
    return loss_te, w

def best_parameters_ridge_regression (y, x, k_fold, lambdas, degrees):
    """Perfom a grid search over lambda and the number of degree to find the combination of these 2
    that gives the lowest error
    return the best degree and best lambda"""
    seed = 12
    k_indices = build_k_indices(y, k_fold, seed)
    best_lambdas = []
    best_losses_te = []
    for degree in degrees:
        losses_te = []
        tx = build_poly(x, degree)
        for lambda_ in lambdas :
            losses_te_tmp = []
            for k in range(k_fold):
                loss_te, w = cross_validation_ridge_regression(y, tx, k_indices, k, lambda_)
                losses_te_tmp.append(loss_te)
            losses_te.append(np.mean(losses_te_tmp))
            
        ind_lambda_opt = np.argmin(losses_te)
        best_lambdas.append(lambdas[ind_lambda_opt])
        best_losses_te.append(losses_te[ind_lambda_opt])
    
    #We select the best hyperparameters (minimizing loss)
    ind_best_degree =  np.argmin(best_losses_te)
    best_degree = degrees[ind_best_degree]
    best_lambda = best_lambdas[ind_best_degree]
    
    return best_degree, best_lambda
