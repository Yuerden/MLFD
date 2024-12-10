import matplotlib.pyplot as plt
import numpy as np
import math
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

# Normalize the features
# Calculate min and max for each feature
symmetry_min = min(symmetries_all)
symmetry_max = max(symmetries_all)
intensity_min = min(intensities_all)
intensity_max = max(intensities_all)

# Calculate shift and scale for each feature
c_symmetry = 2 / (symmetry_max - symmetry_min)
s_symmetry = - (symmetry_max + symmetry_min) / 2
c_intensity = 2 / (intensity_max - intensity_min)
s_intensity = - (intensity_max + intensity_min) / 2

# Apply normalization
symmetries_norm = []
intensities_norm = []
for i in range(len(symmetries_all)):
    sym_norm = c_symmetry * (symmetries_all[i] + s_symmetry)
    int_norm = c_intensity * (intensities_all[i] + s_intensity)
    symmetries_norm.append(sym_norm)
    intensities_norm.append(int_norm)
    
# Verify that features are in [-1, +1]
print("Normalized symmetry range: [{}, {}]".format(min(symmetries_norm), max(symmetries_norm)))
print("Normalized intensity range: [{}, {}]".format(min(intensities_norm), max(intensities_norm)))

# Randomly select 300 data points for dataset D
N = len(labels_all)
indices = list(range(N))
random.shuffle(indices)
train_indices = indices[:300]  # Indices for D
test_indices = indices[300:]   # Indices for Dtest

# Create dataset D
labels_D = [labels_all[i] for i in train_indices]
symmetries_D = [symmetries_norm[i] for i in train_indices]
intensities_D = [intensities_norm[i] for i in train_indices]

# Create test set Dtest (do not use until ready to estimate Eout)
labels_Dtest = [labels_all[i] for i in test_indices]
symmetries_Dtest = [symmetries_norm[i] for i in test_indices]
intensities_Dtest = [intensities_norm[i] for i in test_indices]

# Combine symmetry and intensity features into X and X_test
X_train = np.column_stack((symmetries_D, intensities_D))
X_test = np.column_stack((symmetries_Dtest, intensities_Dtest))

# Convert labels to +1 and -1
y_train = np.array([1 if label == 'o' else -1 for label in labels_D])
y_test = np.array([1 if label == 'o' else -1 for label in labels_Dtest])

# Randomly select 300 data points for dataset D
N = len(labels_all)
indices = list(range(N))
random.shuffle(indices)

train_indices = indices[:300]  # Indices for D
test_indices = indices[300:]   # Indices for Dtest



# So you combined all the data to one data set, normalized the data so that the range of both features is
# [−1, 1] (there is mild data snooping here but, for simplicity, we will live with it) and selected 300 data points
# for your training set and the remaining are a test set.

# K-NN:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))

def knn_predict(train_X, train_y, test_point, k):
    # Compute distances from the test point to all training points
    distances = [(euclidean_distance(test_point, x), y) for x, y in zip(train_X, train_y)]
    # Sort by distance
    distances.sort(key=lambda d: d[0])
    # Take the labels of the k closest points
    k_nearest_labels = [label for _, label in distances[:k]]
    # Return the majority label
    return max(set(k_nearest_labels), key=k_nearest_labels.count)

def loocv_knn(X, y, k):
    errors = 0  # Track total errors
    N = len(X)  # Number of data points
    
    for i in range(N):
        # Leave one out: split data into training and validation
        X_train_minus_one = np.delete(X, i, axis=0)
        y_train_minus_one = np.delete(y, i)
        X_val_one = X[i]
        y_val_one = y[i]
        
        # Predict using k-NN
        pred = knn_predict(X_train_minus_one, y_train_minus_one, X_val_one, k)
        
        # Check if prediction is correct
        if pred != y_val_one:
            errors += 1
    
    # Return average error
    return errors / N

# Find the optimal k
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
k_values = range(1, 20)
loocv_errors = []

for k in k_values:
    error = loocv_knn(X_train_np, y_train_np, k)
    loocv_errors.append(error)

# Optimal k
optimal_k = k_values[np.argmin(loocv_errors)]
print(f"Optimal k: {optimal_k}")

# Plot LOOCV error vs k
plt.plot(k_values, loocv_errors, marker='o')
plt.xlabel("k (Number of neighbors)")
plt.ylabel("LOOCV error")
plt.title("LOOCV Error vs. k (Manual k-NN)")
plt.show()

def plot_decision_boundary(train_X, train_y, k, resolution=200):
    """Plot decision boundary for k-NN."""
    xx, yy = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    Z = np.array([knn_predict(train_X, train_y, [x, y], k) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, edgecolor='k', cmap='coolwarm')
    plt.title(f"Decision Boundary (k={k})")
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.show()

# Plot decision boundary for the optimal k
plot_decision_boundary(X_train_np, y_train_np, optimal_k)

# Compute E_in
errors = 0
for i in range(len(X_train_np)):
    pred = knn_predict(X_train_np, y_train_np, X_train_np[i], optimal_k)
    if pred != y_train_np[i]:
        errors += 1
E_in = errors / len(X_train_np)
print(f"In-sample error (E_in): {E_in:.4f}")

# Cross-validation error for optimal k
E_cv = loocv_knn(X_train_np, y_train_np, optimal_k)
print(f"Cross-validation error (E_cv): {E_cv:.4f}")

# Compute E_test
errors = 0
X_test_np = np.array(X_test)
y_test_np = np.array(y_test)

for i in range(len(X_test_np)):
    pred = knn_predict(X_train_np, y_train_np, X_test_np[i], optimal_k)
    if pred != y_test_np[i]:
        errors += 1
E_test = errors / len(X_test_np)
print(f"Test error (E_test): {E_test:.4f}")

# RBF-Network:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)
X_test_np = np.array(X_test)
y_test_np = np.array(y_test)

def gaussian_rbf(x, centers, r):
    return np.exp(-np.linalg.norm(x - centers, axis=1)**2 / (2 * r**2))

def rbf_network(train_X, train_y, k, r, test_X):
    # Randomly select k centers from training data
    indices = np.random.choice(len(train_X), size=k, replace=False)
    centers = train_X[indices]
    
    # Compute RBF features for training data
    Phi_train = np.array([gaussian_rbf(x, centers, r) for x in train_X])
    
    # Add bias term
    Phi_train = np.hstack((np.ones((Phi_train.shape[0], 1)), Phi_train))
    
    # Compute weights using least squares
    weights = np.linalg.pinv(Phi_train).dot(train_y)
    
    # Compute RBF features for test data
    Phi_test = np.array([gaussian_rbf(x, centers, r) for x in test_X])
    Phi_test = np.hstack((np.ones((Phi_test.shape[0], 1)), Phi_test))
    
    # Predict on test data
    predictions = Phi_test.dot(weights)
    
    # Convert predictions to class labels
    predicted_labels = np.where(predictions >= 0, 1, -1)
    return predicted_labels, weights, centers

def loocv_rbf(X, y, k):
    errors = 0
    N = len(X)
    r = 2 / np.sqrt(k)
    
    for i in range(N):
        # Leave one out
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_val = X[i].reshape(1, -1)
        y_val = y[i]
        
        # Train RBF network
        pred_labels, _, _ = rbf_network(X_train, y_train, k, r, X_val)
        
        # Check if prediction is correct
        if pred_labels[0] != y_val:
            errors += 1
            
    # Return average error
    return errors / N

# Find the optimal k
k_values = range(1, 21)  # You can adjust the range as needed
loocv_errors = []
min_err = 1
optimal_k_rbf = 0

for k in k_values:
    error = loocv_rbf(X_train_np, y_train_np, k)
    loocv_errors.append(error)
    if error < min_err:
        min_err = error
        optimal_k_rbf = k
print(f"Optimal k (number of centers): {optimal_k_rbf}")

# Plot LOOCV error vs k
plt.figure()
plt.plot(k_values, loocv_errors, marker='o')
plt.xlabel("k (Number of centers)")
plt.ylabel("LOOCV error")
plt.title("LOOCV Error vs. k (RBF Network)")
plt.show()

# For the optimal k, compute in-sample error, cross-validation error, and test error

# Set scale r based on optimal k
r_optimal = 2 / np.sqrt(optimal_k_rbf)

# Train on the entire training set
pred_train_labels, weights, centers = rbf_network(X_train_np, y_train_np, optimal_k_rbf, r_optimal, X_train_np)

# Compute in-sample error
errors_in = np.sum(pred_train_labels != y_train_np)
E_in_rbf = errors_in / len(y_train_np)
print(f"In-sample error (E_in): {E_in_rbf:.4f}")

# Cross-validation error for optimal k
E_cv_rbf = loocv_errors[optimal_k_rbf - 1]
print(f"Cross-validation error (E_cv): {E_cv_rbf:.4f}")

# Compute test error
pred_test_labels, _, _ = rbf_network(X_train_np, y_train_np, optimal_k_rbf, r_optimal, X_test_np)
errors_test = np.sum(pred_test_labels != y_test_np)
E_test_rbf = errors_test / len(y_test_np)
print(f"Test error (E_test): {E_test_rbf:.4f}")

# Plot decision boundary for the optimal k
def plot_decision_boundary_rbf(train_X, train_y, centers, weights, r, resolution=200):
    """Plot decision boundary for RBF network."""
    xx, yy = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute RBF features for grid points
    Phi_grid = np.array([gaussian_rbf(x, centers, r) for x in grid_points])
    Phi_grid = np.hstack((np.ones((Phi_grid.shape[0], 1)), Phi_grid))
    
    # Compute predictions
    Z = Phi_grid.dot(weights)
    Z = Z.reshape(xx.shape)
    
    plt.figure()
    plt.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], alpha=0.5, colors=['#FFAAAA','#AAAAFF'])
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, edgecolor='k', cmap='bwr')
    plt.title(f"Decision Boundary (k={optimal_k_rbf})")
    plt.xlabel("Symmetry")
    plt.ylabel("Intensity")
    plt.show()

# Plot decision boundary
plot_decision_boundary_rbf(X_train_np, y_train_np, centers, weights, r_optimal)
