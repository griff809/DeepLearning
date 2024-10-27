
# Lab 1: Linear Regression (corresponding to lecture handout 1)
import numpy as np

# This function computes the polynomial of order 'order' corresponding to a least-squares fit
# to the data (x, y), where 'y' contains the observed values and 'x' contains the x-ordinate
# of each observed value.
#
# The normal equation is sloved in the function 'linear regression'.
def LS_poly(x, y, order, eps = 0):
    # First build the polynomial design matrix (relies only x-ordinates, not observed values)
    X = polynomial_design_matrix(x, order);
    # Then find the polynomial using this matrix and the values 'y'.
    w = linear_regression(X, y, eps=eps);
    return w

# Computes the polynomial design matrix.
#
# For a vector 'x', this contains all powers up to 'order'
# of each element of 'x'.  This kind of matrix is also called
# a Vandermonde matrix.
#
# The numpy array 'x' contains the x-ordinates (x-axis
# values) which we are analyzing.
def polynomial_design_matrix(x, order=1):
    # Create a matrix of zeros, with 'length-of-x' rows and 'order+1' cols
    X = np.zeros(shape=(x.size,order+1))

    # EXERCISE 1: fill the body of this function.
    # See slide 26 of the lecture 1 handout.
    # The exponentiation (power) operator in Python is '**'.
    # Assign to the element (row,col) of a numpy matrix with: M[r,c] = <expression>

    # Hint:
    # Outer loop: iterating over columns; each column gets a higher power
    # for p in range(0, order+1):
    # Inner loop: iterating over rows: each row corresponds to an element of 'x'
    # for i in range(x.size):
    # Element (i,p) of X should be the ith element of 'x' to the power p:

    #X[i,p] = <something>

        # Loop over each column (each power)
    for p in range(order + 1):
        # Loop over each row (each element in x)
        for i in range(x.size):
            # Assign the element i in x to the power p in the matrix
            X[i, p] = x[i] ** p

    return X


# Given values 'y' and the polynomial design matrix for the x-ordinates of those
# values in 'X', find the polynomial having the best fit:
#
# theta = ((X'X + I)^(-1))*X'y
#
# This uses numpy to solve the normal equation (see slide 16 of handout 1)
def linear_regression(X, y, eps=0):
    order = X.shape[1] - 1;
    M = np.dot(X.transpose(), X)

    # EXERCISE 2: implement Tikhonov regularisation.
    # See lecture handout 1, slide 38.
    # print("Eps: " + str(eps))
    #
        # If eps (epsilon) is greater than 0, apply Tikhonov regularization
    if eps > 0:
        # Add eps * I to M, where I is the identity matrix
        I = np.identity(M.shape[0])
        M = M + eps * I
    # <add 'eps' times the identity matrix to M>
    # Hints:
    # There is a function 'identity' in numpy to generate an identity matrix
    # The 'identity' function takes an integer parameter: the size of the (square) identity matrix
    # The shape of a numpy matrix 'A' is accessed with 'A.shape' (no parentheses); this is a tuple
    # The number of rows in a matrix 'A' is then 'A.shape[0]' (or 'len(A)')
    # You can add matrices with '+' -- so you will update 'M' with 'M = M + <amount> * <identity>'
    # Note that the amount of regularization is denoted 'alpha' in the slides but here it's 'eps'.
    theta = np.dot(np.linalg.inv(M), np.dot(X.transpose(), y))
    return theta

# EXERCISE 3: implement computation of mean squared error between two vectors
def mean_squared_error(y1, y2):
    # Compute the elementwise difference
    difference = y1 - y2
    # Square the differences
    squared_difference = difference ** 2
    # Compute the mean of the squared differences
    mse = np.mean(squared_difference)
    return mse


# EXERCISE 4: return the number of the best order for the supplied
# data (see the notebook).
def question_4():
    best_order = 3
    return best_order





# Exercise 5
# EDIT THIS CELL

# This is where need to train a polynomial model for various poylnomial orders
# and regularisation strengths.
#
# eg. poly_model = LS_poly(data_x_train, data_y_train, proposed_order, proposed_eps)
#
# Select the model that has the smallest MSE on the validation set and save it
# as 'best_poly_model'


# Prediction using the learned polynomial coefficients
def predict(x, w):
    X = polynomial_design_matrix(x, len(w) - 1)
    return np.dot(X, w)


# Function to find the best polynomial model
def find_best_model(train_x, train_y, val_x, val_y, max_order=10, eps_range=[0, 0.01, 0.1, 1]):
    best_mse = float('inf')
    best_order = None
    best_eps = None
    best_weights = None

    # Loop over polynomial orders and regularization strengths
    for order in range(1, max_order + 1):
        for eps in eps_range:
            # Generate polynomial design matrix for training data
            X_train = polynomial_design_matrix(train_x, order)
            # Train the model using linear regression with regularization
            w = linear_regression(X_train, train_y, eps)

            # Predict on validation set
            y_pred = predict(val_x, w)

            # Compute validation MSE
            mse = mean_squared_error(val_y, y_pred)

            # Check if this is the best model so far
            if mse < best_mse:
                best_mse = mse
                best_order = order
                best_eps = eps
                best_weights = w
    print(f"Best Polynomial Order: {best_order}, Best Epsilon: {best_eps}, Best MSE: {best_mse}")
    return best_order, best_eps, best_weights

# Example usage
best_order, best_eps, best_weights = find_best_model(data_x_train, data_y_train, data_x_valid, data_y_valid)


best_poly_model = lab_1.LS_poly(data_x_train, data_y_train, best_order,best_eps)