import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import random
from cvxopt import matrix, solvers

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

def polynomial_kernel(X, Z, degree=8):
    return (1 + np.dot(X, Z.T))**degree

def svm_train_dual(X, y, C=1.0, degree=8):
    N = X.shape[0]
    
    # Compute the kernel matrix
    K = polynomial_kernel(X, X, degree=degree)
    
    Y = y.reshape(-1,1)
    Q = (Y * Y.T) * K
    p = -np.ones((N,1))
    
    A = Y.T.astype(float)
    b = np.zeros(1)
    
    G = np.vstack((-np.eye(N), np.eye(N)))
    h = np.hstack((np.zeros(N), np.ones(N)*C))
    
    Q_cvx = matrix(Q, tc='d')
    p_cvx = matrix(p, tc='d')
    G_cvx = matrix(G, tc='d')
    h_cvx = matrix(h, tc='d')
    A_cvx = matrix(A, tc='d')
    b_cvx = matrix(b, tc='d')
    
    sol = solvers.qp(Q_cvx, p_cvx, G_cvx, h_cvx, A_cvx, b_cvx)
    alpha = np.array(sol['x']).flatten()
    
    return alpha

def compute_b(X, y, alpha, degree=8):
    C = np.max(alpha) * 10
    eps = 1e-7
    idx = np.where((alpha > eps) & (alpha < C - eps))[0]
    
    if len(idx) == 0:
        idx = np.where(alpha > eps)[0]
    if len(idx) == 0:
        raise ValueError("No support vectors found to compute b. Check parameters!")
        
    K = polynomial_kernel(X, X[idx], degree=degree)
    y_sv = y[idx]
    al_y = alpha * y
    b_values = []
    for i, k in enumerate(idx):
        b_val = y_sv[i] - np.sum(al_y * K[:,i])
        b_values.append(b_val)
    
    return np.mean(b_values)

def svm_predict(X_train, y_train, alpha, X_test, b, degree=8):
    K = polynomial_kernel(X_test, X_train, degree=degree)
    decision = K.dot((alpha * y_train)) + b
    return np.sign(decision)

small_C = 0.01
large_C = 1000

alpha_small = svm_train_dual(X_train, y_train, C=small_C, degree=8)
b_small = compute_b(X_train, y_train, alpha_small, degree=8)

alpha_large = svm_train_dual(X_train, y_train, C=large_C, degree=8)
b_large = compute_b(X_train, y_train, alpha_large, degree=8)

def plot_decision_boundary_manual_svm(X, y, alpha, b, C, degree=8):
    x_min, x_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    y_min, y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_predict(X, y, alpha, grid_points, b, degree=degree)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.4, levels=[-2,0,2], cmap=plt.cm.bwr)
    
    eps = 1e-7
    sv = np.where(alpha > eps)[0]
    plt.scatter(X[sv,0], X[sv,1], facecolors='none', edgecolors='k', s=100, label='Support Vectors')
    
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.bwr, edgecolors='k', s=50)
    plt.title(f"Decision Boundary (C={C})")
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

# Plot for small C
plot_decision_boundary_manual_svm(X_train, y_train, alpha_small, b_small, small_C, degree=8)

# # Plot for large C
plot_decision_boundary_manual_svm(X_train, y_train, alpha_large, b_large, large_C, degree=8)

def k_fold_cross_validation(X, y, C_values, k=5, degree=8):
    """
    Perform k-fold cross validation to select best C.
    Returns a dictionary {C: average_validation_error}.
    """
    N = X.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    cv_errors = {}
    for C in C_values:
        errors = []
        for i in range(k):
            val_idx = folds[i]
            train_idx = np.hstack([folds[j] for j in range(k) if j != i])
            
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            alpha = svm_train_dual(X_tr, y_tr, C=C, degree=degree)
            b = compute_b(X_tr, y_tr, alpha, degree=degree)

            y_pred_val = svm_predict(X_tr, y_tr, alpha, X_val, b, degree=degree)
            val_error = np.mean(y_pred_val != y_val)
            errors.append(val_error)
        cv_errors[C] = np.mean(errors)
    return cv_errors

def plot_decision_boundary_manual_svm(X, y, alpha, b, C, degree=8):
    x_min, x_max = X[:,0].min()-0.1, X[:,0].max()+0.1
    y_min, y_max = X[:,1].min()-0.1, X[:,1].max()+0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = svm_predict(X, y, alpha, grid_points, b, degree=degree)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6,5))
    plt.contourf(xx, yy, Z, alpha=0.4, levels=[-2,0,2], cmap=plt.cm.bwr)
    
    eps = 1e-7
    sv = np.where(alpha > eps)[0]
    plt.scatter(X[sv,0], X[sv,1], facecolors='none', edgecolors='k', s=100, label='Support Vectors')
    
    plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.bwr, edgecolors='k', s=50)
    plt.title(f"Decision Boundary (C={C})")
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.legend()
    plt.show()

C_values = np.logspace(-2, 3, 10)  # from 0.01 to 1000 in a log scale

cv_errors = k_fold_cross_validation(X_train, y_train, C_values, k=5, degree=8)
for C, err in cv_errors.items():
    print(f"C={C}, CV Error={err}")

best_C = min(cv_errors, key=cv_errors.get)
print("Best C found by CV:", best_C)

alpha_best = svm_train_dual(X_train, y_train, C=best_C, degree=8)
b_best = compute_b(X_train, y_train, alpha_best, degree=8)

plot_decision_boundary_manual_svm(X_train, y_train, alpha_best, b_best, best_C, degree=8)

y_pred_test = svm_predict(X_train, y_train, alpha_best, X_test, b_best, degree=8)
test_error = np.mean(y_pred_test != y_test)
print("Test Error with best_C:", test_error)