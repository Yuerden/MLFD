import numpy as np
import matplotlib.pyplot as plt
import random

# Initialize empty lists to hold all data
labels_all = []
symmetries_all = []
intensities_all = []

# Function to process a file and extract features
def process_file(filename):
    with open(filename, "r") as file:
        for line in file:
            # Split the line into label and pixel values
            data = line.strip().split()
            label = data[0]

            # Process all digits                        #'1' and '5'
            if True:                                    #label == "1.0000" or label == "5.0000":
                # Convert pixel values to floats
                values = list(map(float, data[1:]))

                # Append the label
                if label == "1.0000":
                    labels_all.append('o')  # You can use 1 or 'o' for '1'
                else:
                    labels_all.append('x')  # You can use -1 or 'x' for '[2,...,9,0]'

                # Symmetry calculation
                cur_sym = 0
                for i in range(16):  # For each row
                    for j in range(8):  # For half the columns
                        left_pixel = values[i*16 + j]
                        right_pixel = values[i*16 + (15 - j)]
                        cur_sym += abs(left_pixel - right_pixel)
                symmetries_all.append(cur_sym / 256)

                # Intensity calculation
                cur_intense = sum(values)
                intensities_all.append(cur_intense / 256)

# Process both training and test files
process_file("ZipDigits.train")
process_file("ZipDigits.test")

# Randomly select 300 data points for dataset D
N = len(labels_all)
indices = list(range(N))
random.shuffle(indices)
train_indices = indices[:300]  # Indices for D
test_indices = indices[300:]   # Indices for Dtest

# Create dataset D
labels_D = [labels_all[i] for i in train_indices]
symmetries_D = [symmetries_all[i] for i in train_indices]
intensities_D = [intensities_all[i] for i in train_indices]

# Create test set Dtest (do not use until ready to estimate Eout)
labels_Dtest = [labels_all[i] for i in test_indices]
symmetries_Dtest = [symmetries_all[i] for i in test_indices]
intensities_Dtest = [intensities_all[i] for i in test_indices]

# Combine symmetry and intensity features into X and X_test
X_train = np.column_stack((symmetries_D, intensities_D))
X_test = np.column_stack((symmetries_Dtest, intensities_Dtest))

# Convert labels to +1 and -1
y_train = np.array([1 if label == 'o' else -1 for label in labels_D])
y_test = np.array([1 if label == 'o' else -1 for label in labels_Dtest])

# Neural network parameters
d_in = 2
m = 10
d_out = 1

def tanh(s):
    return np.tanh(s)

def tanh_deriv(s):
    return 1 - np.tanh(s)**2

def identity(s):
    return s

def identity_deriv(s):
    return np.ones_like(s)

def sign_func(s):
    return np.where(s >= 0, 1, -1)

def forward_pass(X, W1, W2, output_activation='identity'):
    N = X.shape[0]
    Xb = np.hstack((np.ones((N,1)), X)) # (N, d_in+1)

    Z_h = Xb.dot(W1)  # (N, m)
    A_h = tanh(Z_h)   # hidden activations (N,m)

    A_hb = np.hstack((np.ones((N,1)), A_h)) # (N, m+1)
    Z_o = A_hb.dot(W2)  # (N, d_out)
    if output_activation == 'tanh':
        A_o = tanh(Z_o)
    elif output_activation == 'identity':
        A_o = identity(Z_o)
    else:
        raise ValueError("Unknown output activation")

    return Xb, Z_h, A_h, A_hb, Z_o, A_o

def compute_error(y_pred, y):
    return np.mean((y_pred - y.reshape(-1,1))**2) / 4.0

def backprop(X, y, W1, W2, output_activation='identity', lambda_reg=0.0):
    Xb, Z_h, A_h, A_hb, Z_o, A_o = forward_pass(X, W1, W2, output_activation=output_activation)

    if output_activation == 'identity':
        dA_o = (A_o - y.reshape(-1,1)) / (2.0)  # dE/dA_o since E = (1/(4N))*sum(...) 
        dZ_o = dA_o # derivative of identity is 1
    elif output_activation == 'tanh':
        dA_o = (A_o - y.reshape(-1,1)) / (2.0)
        dZ_o = dA_o * tanh_deriv(Z_o)
    else:
        raise ValueError("Only 'identity' or 'tanh' supported here for training")

    G2 = (A_hb.T.dot(dZ_o)) / X.shape[0]
    G2[1:] += (lambda_reg/X.shape[0])*W2[1:]

    dA_h = dZ_o.dot(W2[1:].T) # remove bias from W2
    dZ_h = dA_h * tanh_deriv(Z_h)

    G1 = (Xb.T.dot(dZ_h)) / X.shape[0]
    G1[1:,:] += (lambda_reg/X.shape[0])*W1[1:,:]

    return G1, G2, A_o

def init_weights(d_in, m, d_out):
    W1 = 0.01*np.random.randn(d_in+1, m)  # (3 x 10)
    W2 = 0.01*np.random.randn(m+1, d_out) # (11 x 1)
    return W1, W2

def variable_lr_gd(X, y, max_iter=2_000_000, initial_eta=0.1, eta_decay=0.999999, lambda_reg=0.0):
    W1, W2 = init_weights(d_in, m, d_out)
    errors = []

    eta = initial_eta
    for t in range(max_iter):
        G1, G2, A_o = backprop(X, y, W1, W2, 'identity', lambda_reg=lambda_reg)
        W1 -= eta * G1
        W2 -= eta * G2

        eta *= eta_decay

        if t % 10000 == 0:
            err = compute_error(A_o, y)
            errors.append((t, err))
            print("Iter:", t, "Error:", err)

    return W1, W2, errors

def sgd(X, y, max_iter=20_000_000, eta=0.001, lambda_reg=0.0):
    W1, W2 = init_weights(d_in, m, d_out)
    errors = []
    N = X.shape[0]

    for t in range(max_iter):
        i = np.random.randint(N)
        X_i = X[i:i+1,:]
        y_i = y[i:i+1]

        G1, G2, A_o = backprop(X_i, y_i, W1, W2, 'identity', lambda_reg=lambda_reg)
        W1 -= eta * G1
        W2 -= eta * G2

        if t % 100000 == 0:
            A_o_full = forward_pass(X, W1, W2, 'identity')[-1]
            err = compute_error(A_o_full, y)
            errors.append((t/N, err))  # plot error vs iteration/N
            print("Iter:", t, "Iteration/N:", t/N, "Error:", err)

    return W1, W2, errors

def predict(X, W1, W2):
    _, _, _, A_hb, Z_o, A_o = forward_pass(X, W1, W2, output_activation='identity')
    return sign_func(A_o)

def plot_decision_boundary(W1, W2, X, y, title="Decision Boundary"):
    x_min, x_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    y_min, y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    preds = predict(grid_points, W1, W2).reshape(xx.shape)

    plt.contourf(xx, yy, preds, alpha=0.5, levels=[-2,0,2], cmap=plt.cm.bwr)
    # Plot training points
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.bwr, edgecolors='k')
    plt.title(title)
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.show()

# (a) Variable learning rate gradient descent
W1_gd, W2_gd, errors_gd = variable_lr_gd(X_train, y_train, max_iter=2_000_000)
iters_gd = [e[0] for e in errors_gd]
errs_gd = [e[1] for e in errors_gd]
plt.plot(iters_gd, errs_gd)
plt.title("Ein vs Iteration (Variable LR GD)")
plt.xlabel("Iteration #")
plt.ylabel("Ein")
plt.show()
plot_decision_boundary(W1_gd, W2_gd, X_train, y_train, title="Decision Boundary (Variable LR GD)")

# (b) Stochastic Gradient Descent
W1_sgd, W2_sgd, errors_sgd = sgd(X_train, y_train, max_iter=20_000_000, eta=0.0001)
iters_sgd = [e[0] for e in errors_sgd]  # iteration/N
errs_sgd = [e[1] for e in errors_sgd]
plt.plot(iters_sgd, errs_sgd)
plt.title("Ein vs (Iteration/N) (SGD)")
plt.xlabel("Iteration/N")
plt.ylabel("Ein")
plt.show()
plot_decision_boundary(W1_sgd, W2_sgd, X_train, y_train, title="Decision Boundary (SGD)")

# (c) Run with weight decay:
lambda_reg = 0.01/X_train.shape[0]
print("Running Variable LR GD with weight decay...")
W1_gd_decay, W2_gd_decay, errors_gd_decay = variable_lr_gd(X_train, y_train, max_iter=2_000_000, lambda_reg=lambda_reg)

iters_gd_decay = [e[0] for e in errors_gd_decay]
errs_gd_decay = [e[1] for e in errors_gd_decay]
plt.plot(iters_gd_decay, errs_gd_decay)
plt.title("Ein vs Iteration (Variable LR GD) with Weight Decay")
plt.xlabel("Iteration #")
plt.ylabel("Ein")
plt.show()

plot_decision_boundary(W1_gd_decay, W2_gd_decay, X_train, y_train, title="Decision Boundary (Variable LR GD + Weight Decay)")

print("Running SGD with weight decay...")
W1_sgd_decay, W2_sgd_decay, errors_sgd_decay = sgd(X_train, y_train, max_iter=20_000_000, eta=0.0001, lambda_reg=lambda_reg)
iters_sgd_decay = [e[0] for e in errors_sgd_decay]
errs_sgd_decay = [e[1] for e in errors_sgd_decay]
plt.plot(iters_sgd_decay, errs_sgd_decay)
plt.title("Ein vs (Iteration/N) (SGD) with Weight Decay")
plt.xlabel("Iteration/N")
plt.ylabel("Ein")
plt.show()

plot_decision_boundary(W1_sgd_decay, W2_sgd_decay, X_train, y_train, title="Decision Boundary (SGD + Weight Decay)")


# (d) Early Stopping
def variable_lr_gd_early_stopping(X, y, X_val, y_val, max_iter=2_000_000, initial_eta=0.1, eta_decay=0.999999, lambda_reg=0.0, check_freq=10000, patience_limit=5):
    W1, W2 = init_weights(d_in, m, d_out)
    best_W1, best_W2 = W1.copy(), W2.copy()
    best_val_err = np.inf
    patience_counter = 0
    eta = initial_eta
    errors = []

    for t in range(max_iter):
        G1, G2, A_o_tr = backprop(X, y, W1, W2, 'identity', lambda_reg=lambda_reg)
        W1 -= eta * G1
        W2 -= eta * G2

        eta *= eta_decay

        if t % check_freq == 0:
            A_o_val = forward_pass(X_val, W1, W2, 'identity')[-1]
            val_err = compute_error(A_o_val, y_val)
            A_o_full = forward_pass(X, W1, W2, 'identity')[-1]
            tr_err = compute_error(A_o_full, y)
            errors.append((t, tr_err, val_err))
            print(f"Variable LR GD - Iter: {t}, Training Error: {tr_err}, Validation Error: {val_err}")

            if val_err < best_val_err:
                best_val_err = val_err
                best_W1, best_W2 = W1.copy(), W2.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience_limit:
                    print("Early stopping for Variable LR GD at iteration", t)
                    break

    return best_W1, best_W2, errors

def sgd_early_stopping(X, y, X_val, y_val, max_iter=20_000_000, eta=0.001, lambda_reg=0.0, check_freq=1000000, patience_limit=5):
    W1, W2 = init_weights(d_in, m, d_out)
    best_W1, best_W2 = W1.copy(), W2.copy()
    best_val_err = np.inf
    patience_counter = 0
    N = X.shape[0]

    errors = []

    for t in range(max_iter):
        i = np.random.randint(N)
        X_i = X[i:i+1,:]
        y_i = y[i:i+1]
        G1, G2, A_o = backprop(X_i, y_i, W1, W2, 'identity', lambda_reg=lambda_reg)
        W1 -= eta * G1
        W2 -= eta * G2

        if t % check_freq == 0 and t > 0:
            A_o_val = forward_pass(X_val, W1, W2, 'identity')[-1]
            val_err = compute_error(A_o_val, y_val)
            A_o_full = forward_pass(X, W1, W2, 'identity')[-1]
            tr_err = compute_error(A_o_full, y)
            errors.append((t/N, tr_err, val_err))
            print(f"SGD - Iter: {t}, Iter/N: {t/N}, Training Error: {tr_err}, Validation Error: {val_err}")

            if val_err < best_val_err:
                best_val_err = val_err
                best_W1, best_W2 = W1.copy(), W2.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience_limit:
                    print("Early stopping for SGD at iteration", t)
                    break

    return best_W1, best_W2, errors


# Part (d):
X_val = X_train[:50]
y_val = y_train[:50]
X_tr = X_train[50:]
y_tr = y_train[50:]

W1_gd_es, W2_gd_es, errors_gd_es = variable_lr_gd_early_stopping(X_tr, y_tr, X_val, y_val, max_iter=2_000_000, initial_eta=0.1, eta_decay=0.999999)

iters_gd_es = [e[0] for e in errors_gd_es]
tr_err_gd_es = [e[1] for e in errors_gd_es]
val_err_gd_es = [e[2] for e in errors_gd_es]

plt.figure()
plt.plot(iters_gd_es, tr_err_gd_es, label="Training Error")
plt.plot(iters_gd_es, val_err_gd_es, label="Validation Error")
plt.title("Ein vs Iteration (Variable LR GD with Early Stopping)")
plt.xlabel("Iteration #")
plt.ylabel("Error")
plt.legend()
plt.show()  # Figure 1

plt.figure()
plot_decision_boundary(W1_gd_es, W2_gd_es, X_train, y_train, title="Decision Boundary (Variable LR GD + Early Stopping)")  


W1_sgd_es, W2_sgd_es, errors_sgd_es = sgd_early_stopping(X_tr, y_tr, X_val, y_val, max_iter=20_000_000, eta=0.0001)

iters_sgd_es = [e[0] for e in errors_sgd_es]  # iteration/N
tr_err_sgd_es = [e[1] for e in errors_sgd_es]
val_err_sgd_es = [e[2] for e in errors_sgd_es]

plt.figure()
plt.plot(iters_sgd_es, tr_err_sgd_es, label="Training Error")
plt.plot(iters_sgd_es, val_err_sgd_es, label="Validation Error")
plt.title("Ein vs (Iteration/N) (SGD with Early Stopping)")
plt.xlabel("Iteration/N")
plt.ylabel("Error")
plt.legend()
plt.show()  # Figure 3

plt.figure()
plot_decision_boundary(W1_sgd_es, W2_sgd_es, X_train, y_train, title="Decision Boundary (SGD + Early Stopping)")  