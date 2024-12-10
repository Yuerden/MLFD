import numpy as np
import matplotlib.pyplot as plt

# Parameters
thk = 5
rad = 10
sep = 5       # separation between the semi-circles
inner = rad  # inner radius
outer = rad + thk  # outer radius
num_points = 2000  # number of points per class

# Generate semi-circle
def generate_semi(rad, thk, sep, num_points, label):
    origin_x = thk + rad
    origin_y = thk + rad + sep
    theta = np.pi * np.random.rand(num_points)  # angles between 0 and pi
    if label == -1:  # bottom semi-circle
        origin_x += thk / 2 + rad
        origin_y -= sep
        theta = np.pi + np.pi * np.random.rand(num_points)
    r = (outer - inner) * np.random.rand(num_points) + inner  # radii between inner and outer
    x = origin_x + r * np.cos(theta)
    y = origin_y + r * np.sin(theta)
    labels = [label] * num_points
    return x, y, labels

# Perceptron algorithm
def perceptron(X, y, max_iterations=10000):
    w = np.zeros(X.shape[1])  # Initialize weight vector
    iterations = 0
    for _ in range(max_iterations):
        misclassified = 0
        for i in range(X.shape[0]):
            if np.sign(np.dot(w, X[i])) != y[i]:
                w += y[i] * X[i]
                misclassified += 1
        iterations += 1
        if misclassified == 0:
            break
    return w, iterations

# Linear Regression function using least squares
def linear_regression(X, y):
    # Closed-form solution (w = (X^T X)^(-1) X^T y)
    w = np.linalg.inv(X.T @ X) @ X.T @ y
    return w

# Generate semi-circles
x_top, y_top, labels_top = generate_semi(rad, thk, sep, num_points // 2, +1)
x_bottom, y_bottom, labels_bottom = generate_semi(rad, thk, sep, num_points // 2, -1)

# Combine the data for both semi-circles
x = np.append(x_top, x_bottom)
y = np.append(y_top, y_bottom)
labels = np.array(labels_top + labels_bottom)

# Prepare data for Perceptron (add bias term)
X = np.c_[np.ones(x.shape[0]), x, y]  # Add bias term as the first column
y = labels  # labels: +1 or -1

# Train perceptron or regression
# w, iterations = perceptron(X, y)              #3.1a
w = linear_regression(X, y)                     #3.1b

# Plot the semi-circles
plt.figure(figsize=(8, 6))
plt.scatter(x_top, y_top, c='blue', label='+1', alpha=0.5)
plt.scatter(x_bottom, y_bottom, c='red', label='-1', alpha=0.5)

# Plot the decision boundary (w0 + w1*x + w2*y = 0)
x_vals = np.linspace(min(x), max(x), 100)
y_vals = -(w[0] + w[1] * x_vals) / w[2]  # Solve for y in terms of x
plt.plot(x_vals, y_vals, 'k-', label='Decision Boundary')

# plt.title(f'Semi-Circle Data with Perceptron (Iterations: {iterations})') #3.1a
plt.title('Semi-Circle Data with Linear Regression')                        #3.1b
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
