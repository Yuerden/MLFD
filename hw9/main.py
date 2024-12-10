import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn.model_selection import KFold

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




# Problem 1: 8th Order Feature Transform
def legendre_polynomials(x, order):
    L = [np.ones_like(x), x.copy()]
    for n in range(2, order + 1):
        Ln = ((2 * n - 1) * x * L[n - 1] - (n - 1) * L[n - 2]) / n
        L.append(Ln)
    return L  # Returns list of Legendre polynomials up to specified order

def construct_feature_matrix(X, max_order):
    x1 = X[:, 0]
    x2 = X[:, 1]
    Lx1 = legendre_polynomials(x1, max_order)
    Lx2 = legendre_polynomials(x2, max_order)
    features = []
    for n_total in range(max_order + 1):
        for n1 in range(n_total + 1):
            n2 = n_total - n1
            feature = Lx1[n1] * Lx2[n2]
            features.append(feature)
    Z = np.column_stack(features)
    return Z

# Construct Z for training and test data
max_order = 8
Z_train = construct_feature_matrix(X_train, max_order)
Z_test = construct_feature_matrix(X_test, max_order)

# Dimensions of Z
print("Dimensions of Z_train:", Z_train.shape)  # Should be (N, 45)
print("Dimensions of Z_test:", Z_test.shape)

# Problem 2: Overfitting (λ = 0)
lambda_0 = 0
I = np.identity(Z_train.shape[1])
w_reg_0 = np.linalg.inv(Z_train.T @ Z_train + lambda_0 * I) @ Z_train.T @ y_train

# Plot decision boundary
def plot_decision_boundary(Z, w, X, y, title):
    # Create a grid of points
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    Z_grid = construct_feature_matrix(grid, max_order)
    zz = Z_grid @ w
    zz = zz.reshape(xx.shape)
    plt.contourf(xx, yy, zz, levels=[-np.inf, 0, np.inf], alpha=0.3, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
    plt.title(title)
    plt.xlabel('Symmetry')
    plt.ylabel('Intensity')
    plt.show()

plot_decision_boundary(Z_train, w_reg_0, X_train, y_train, 'Decision Boundary (λ = 0)')

# Problem 3: Regularization (λ = 2)
lambda_2 = 2
w_reg_2 = np.linalg.inv(Z_train.T @ Z_train + lambda_2 * I) @ Z_train.T @ y_train

plot_decision_boundary(Z_train, w_reg_2, X_train, y_train, 'Decision Boundary (λ = 2)')

# Problem 4: Cross Validation
lambdas = np.arange(0, 10.01, 0.01)
E_cv = []
E_test = []

ZTZ = Z_train.T @ Z_train
ZTy = Z_train.T @ y_train
N = Z_train.shape[0]

for lam in lambdas:
    w_reg = np.linalg.inv(ZTZ + lam * I) @ ZTy
    y_pred = Z_train @ w_reg
    errors = y_train - y_pred
    H = Z_train @ np.linalg.inv(ZTZ + lam * I) @ Z_train.T
    h = np.diag(H)
    E_cv_lam = np.mean((errors / (1 - h)) ** 2)
    E_cv.append(E_cv_lam)
    # Test error
    y_test_pred = Z_test @ w_reg
    E_test_lam = np.mean((y_test - y_test_pred) ** 2)
    E_test.append(E_test_lam)

# Plot E_cv(λ) and E_test(λ)
plt.plot(lambdas, E_cv, label='E_cv(λ)')
plt.plot(lambdas, E_test, label='E_test(λ)')
plt.xlabel('λ')
plt.ylabel('Error')
plt.title('Cross Validation Error and Test Error vs λ')
plt.legend()
plt.ylim(0, .25)
plt.show()

# Problem 5: Pick λ*
lambda_star = lambdas[np.argmin(E_cv)]
print("Best λ (lambda*):", lambda_star)
w_reg_star = np.linalg.inv(ZTZ + lambda_star * I) @ ZTy

plot_decision_boundary(Z_train, w_reg_star, X_train, y_train, f'Decision Boundary (λ* = {lambda_star})')

# Problem 6: Estimate the Classification Error E_out(w_reg(λ*))
# Classification error on test set
y_test_pred_class = np.sign(Z_test @ w_reg_star)
E_test_classification = np.mean(y_test != y_test_pred_class)
print("Classification Error on Test Set:", E_test_classification)

# 99% Confidence Interval
from statsmodels.stats.proportion import proportion_confint

lower, upper = proportion_confint(count=int(E_test_classification * len(y_test)),
                                  nobs=len(y_test),
                                  alpha=0.01,
                                  method='normal')
print(f"99% Confidence Interval for E_out: [{lower}, {upper}]")