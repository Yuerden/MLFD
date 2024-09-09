import numpy as np
import matplotlib.pyplot as plt

# Initialize target function f cooresponding to the weights
def initialize_f(dim=2):
    # Generate a random separating hyperplane with dim+1
    f = np.random.uniform(-1, 1, dim + 1)
    return f

# Helper function to generate a linearly separable dataset
def generate_linearly_separable_data(n, target_f, dim=2):
    X = np.random.uniform(-1, 1, (n, dim))
    # Add bias term to X (column of 1s)
    X_bias = np.c_[np.ones(n), X]
    # Assign labels based on the sign of the dot product with the true weights
    y = np.sign(X_bias.dot(target_f))
    return X, y

# Perceptron Learning Algorithm (PLA)
def perceptron_learning_algorithm(X, y, max_iters=1000):
    n, dim = X.shape
    X_bias = np.c_[np.ones(n), X]  # Add bias term (column of 1s)
    h = np.zeros(dim + 1)  # Initialize weight vector with zeros
    updates = 0

    for _ in range(max_iters):
        misclassified = False
        for i in range(n):
            if np.sign(h.dot(X_bias[i])) != y[i]:
                h += y[i] * X_bias[i]  # Update the weight vector
                updates += 1
                misclassified = True
        if not misclassified:  # Converged if no misclassified points
            break

    return h, updates

# Plotting function to visualize the dataset and decision boundary
def plot_dataset_and_target(X, y, target_f, title):
    plt.figure()
    # Plot the data points
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], color='blue', marker='o')
        else:
            plt.scatter(X[i, 0], X[i, 1], color='red', marker='x')

    # Plot the decision boundary
    x1 = np.linspace(-1, 1, 100)
    x2 = -(target_f[0] + target_f[1] * x1) / target_f[2]
    plt.plot(x1, x2, color='green', label='Target Function f')
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plotting function to visualize the dataset and decision boundary
def plot_dataset_and_boundary(X, y, target_f, g, title):
    plt.figure()
    # Plot the data points
    for i in range(len(y)):
        if y[i] == 1:
            plt.scatter(X[i, 0], X[i, 1], color='blue', marker='o')
        else:
            plt.scatter(X[i, 0], X[i, 1], color='red', marker='x')

    # Plot the decision boundary
    x1 = np.linspace(-1, 1, 100)
        # Plot Target
    x2 = -(target_f[0] + target_f[1] * x1) / target_f[2]
    plt.plot(x1, x2, color='green', label='Target Function f')
        # Plot Final Hypothesis g
    x3 = -(g[0] + g[1] * x1) / g[2]
    plt.plot(x1, x3, color='orange', label='Final Hypothesis g')
    
    plt.title(title)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True)
    plt.show()

# (a) Gen 20, Plot examples, target function f
    # Generate a linearly separable dataset of size 20
target_f = initialize_f()
X_20, y_20 = generate_linearly_separable_data(20, target_f)
    # Plot examples, target function f
plot_dataset_and_target(X_20, y_20, target_f, 'Linearly Separable Data of Size 20')

# (b) old 20
    # Run the perceptron algorithm on data set (a)
g, updates_20 = perceptron_learning_algorithm(X_20, y_20)
    # Plot examples, target function f, final hypothesis g, report number of updates to converge
plot_dataset_and_boundary(X_20, y_20, target_f, g, f'PLA for Size 20, Updates = {updates_20}')

# (c) new 20
    # New randomly generated dataset of size 20
X_20_new, y_20_new = generate_linearly_separable_data(20, target_f)
    # Run the perceptron algorithm on data set (c)
g, updates_20_new = perceptron_learning_algorithm(X_20_new, y_20_new)
    # Plot examples, target function f, final hypothesis g, report number of updates to converge
plot_dataset_and_boundary(X_20_new, y_20_new, target_f, g, f'New PLA for Size 20, Updates = {updates_20_new}')

# (d) 100
    # New randomly generated dataset of size 100
X_100, y_100 = generate_linearly_separable_data(100, target_f)
    # Run the perceptron algorithm on data set (d)
g, updates_100 = perceptron_learning_algorithm(X_100, y_100)
    # Plot examples, target function f, final hypothesis g, report number of updates to converge
plot_dataset_and_boundary(X_100, y_100, target_f, g, f'PLA for Size 100, Updates = {updates_100}')

# (e) 1000
    # New randomly generated dataset of size 1000
X_1000, y_1000 = generate_linearly_separable_data(1000, target_f)
    # Run the perceptron algorithm on data set (e)
g, updates_1000 = perceptron_learning_algorithm(X_1000, y_1000)
    # Plot examples, target function f, final hypothesis g, report number of updates to converge
plot_dataset_and_boundary(X_1000, y_1000, target_f, g, f'PLA for Size 1000, Updates = {updates_1000}')
