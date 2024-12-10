import matplotlib.pyplot as plt
import numpy as np
import math

# Open the training file
with open("zip.train", "r") as train:
    labels = []
    symmetries = []
    intensities = []

    for line in train:
        # Split the line into label and pixel values
        data = line.strip().split()
        label = data[0]

        # Process only digits '1' and '5'
        if label == "1.0000" or label == "5.0000":
            # Convert pixel values to floats
            values = list(map(float, data[1:]))

            # Append the label
            if label == "1.0000":
                labels.append('o')
            else:  # label == "5.0000"
                labels.append('x')

            # Symmetry calculation
            cur_sym = 0
            for i in range(16):  # For each row
                for j in range(8):  # For half the columns
                    left_pixel = values[i*16 + j]
                    right_pixel = values[i*16 + (15 - j)]
                    cur_sym += abs(left_pixel - right_pixel)
            symmetries.append(cur_sym / 256)

            # Intensity calculation
            cur_intense = sum(values)
            intensities.append(cur_intense / 256)

# Open the test file
with open("zip.test", "r") as test:
    labels_t = []
    symmetries_t = []
    intensities_t = []

    for line in test:
        # Split the line into label and pixel values
        data = line.strip().split()
        label = data[0]

        # Process only digits '1' and '5'
        if label == "1.0000" or label == "5.0000":
            # Convert pixel values to floats
            values = list(map(float, data[1:]))

            # Append the label
            if label == "1.0000":
                labels_t.append('o')
            else:  # label == "5.0000"
                labels_t.append('x')

            # Symmetry calculation
            cur_sym = 0
            for i in range(16):  # For each row
                for j in range(8):  # For half the columns
                    left_pixel = values[i*16 + j]
                    right_pixel = values[i*16 + (15 - j)]
                    cur_sym += abs(left_pixel - right_pixel)
            symmetries_t.append(cur_sym / 256)

            # Intensity calculation
            cur_intense = sum(values)
            intensities_t.append(cur_intense / 256)

# Initialize y and X and w
y = np.array([1 if label == 'o' else -1 for label in labels])  # +1 for 'o' and -1 for 'x'
X = np.array([[1, symmetries[i], intensities[i]] for i in range(len(labels))])  # Add bias term
y_test = np.array([1 if label == 'o' else -1 for label in labels_t])  # +1 for 'o' and -1 for 'x'
X_test = np.array([[1, symmetries_t[i], intensities_t[i]] for i in range(len(labels_t))])  # Add bias term
starting_weights = np.array([0, 0, 0])  # Ensure this is a NumPy array
print(f"N_train: {len(y)} and N_test: {len(y_test)}")

def plot_line(weights, title, train=True):
    if train:
        for i in range(len(symmetries)):
            color = 'blue' if labels[i] == 'o' else 'red'
            plt.scatter(symmetries[i], intensities[i], marker=labels[i], c=color)

        x_values = np.linspace(min(symmetries), max(symmetries), 100)
    else:
        for i in range(len(symmetries_t)):
            color = 'blue' if labels_t[i] == 'o' else 'red'
            plt.scatter(symmetries_t[i], intensities_t[i], marker=labels_t[i], c=color)

        x_values = np.linspace(min(symmetries_t), max(symmetries_t), 100)
    
    y_values = -(weights[0] + weights[1] * x_values) / weights[2]
    
    plt.plot(x_values, y_values, color='green', linestyle='-', linewidth=2, label="Decision Boundary")
    plt.ylim(-1.2, 1) 
    plt.title(f'{title}')
    plt.xlabel('Symmetry')
    plt.ylabel('Average Intensity')
    plt.legend()
    plt.show()
def sign(number):
    if(number >= 0):
        return 1
    return -1
def calculate_Ein_Etest(X,X_test,w,method):
    train_error = 0
    test_error = 0
    for i in range(len(y)):
        if(y[i] != sign(np.dot(X[i], w))): train_error+=1
    for i in range(len(y_test)):
        if(y_test[i] != sign(np.dot(X_test[i], w))): test_error+=1

    print(f"{method}:::::::")
    print(f"    E_in: {train_error / len(y)}")
    print(f"    E_test: {test_error / len(y_test)}\n")

# (i) Linear Regression for classification followed by pocket for improvement.
def accuracy_score(actual, predicted):
    assert len(actual) == len(predicted)
    correct = 0
    for i in range(len(actual)):
        if actual[i]==predicted[i]: correct += 1
    return correct/len(actual)
def linear_regression_classification(X, y):
    weights = np.linalg.pinv(X.T @ X) @ X.T @ y
    return weights
def pocket_algorithm(X, y, weights, max_iter=1000):
    best_weights = weights
    best_accuracy = accuracy_score(y, np.sign(X @ weights))
    
    for _ in range(max_iter):
        predictions = np.sign(X @ weights)
        misclassified = np.where(predictions != y)[0]
        
        if len(misclassified) == 0:
            break

        i = np.random.choice(misclassified)
        weights += y[i] * X[i]
        
        accuracy = accuracy_score(y, np.sign(X @ weights))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights.copy()
    
    return best_weights
w_i = starting_weights.copy()
w_i = linear_regression_classification(X,y)
w_i = pocket_algorithm(X,y,w_i)
plot_line(w_i, "Linear Regression with Pocket Train")
plot_line(w_i, "Linear Regression with Pocket Test", train=False)
calculate_Ein_Etest(X,X_test,w_i, "Linear Regression with Pocket")

# (ii) Logistic regression for classification using gradient descent.
def gradient_descent(X, y, weights, learning_rate=0.1, max_iter=100000):
    N = len(X)  # Number of samples
    weights = weights.astype(float)  # Ensure weights are float

    for _ in range(max_iter):
        vt = np.zeros_like(weights, dtype=float)  # Initialize vt as float
        for i in range(N):
            xi = X[i]
            yi = y[i]
            
            # Calculate vt as the gradient for this example
            vt += (yi * xi) / (1 + math.pow(math.e, yi * (np.dot(weights, xi))))
        
        # Update weights with the average gradient
        weights += learning_rate * vt / N

    return weights
w_ii = starting_weights.copy()
w_ii = gradient_descent(X, y, w_ii)
plot_line(w_ii, "Gradient Descent Train")
plot_line(w_ii, "Gradient Descent Test", train=False)
calculate_Ein_Etest(X,X_test,w_ii, "Gradient Descent")

# (iii) Logistic regression for classification using stochastic gradient descent.
def stochastic_gradient_descent(X, y, weights, learning_rate=0.1, max_iter=100000):
    N = len(X)  # Number of samples
    weights = weights.astype(float)  # Ensure weights are float

    for _ in range(max_iter):
        for i in range(N):
            xi = X[i]
            yi = y[i]
            
            # Calculate the gradient for the current example
            gradient = (yi * xi) / (1 + np.exp(yi * np.dot(xi, weights)))
            
            # Update weights based on the gradient of the current sample
            weights += learning_rate * gradient

    return weights
w_iii = starting_weights.copy()
w_iii = stochastic_gradient_descent(X, y, w_iii)
plot_line(w_iii, "Stochastic Gradient Descent Train")
plot_line(w_iii, "Stochastic Gradient Descent Test", train=False)
calculate_Ein_Etest(X,X_test,w_iii, "Stochastic Gradient Descent")

# (iv) Linear Programming for classification (Graduate, 6xxx-level, only).
def linear_programming_classification(X, y, learning_rate=0.1, max_iter=1000):
    # Initialize weights as small random values
    weights = np.random.randn(X.shape[1]) * learning_rate
    
    for _ in range(max_iter):
        for i in range(len(y)):
            xi = X[i]
            yi = y[i]
            constraint = yi * np.dot(weights, xi) # y*(w*x) >= 1
            
            # If constraint is violated
            if constraint < 1:
                weights += learning_rate * yi * xi
            else:
                weights -= learning_rate * weights * 0.01

    return weights
w_iv = linear_programming_classification(X, y)
plot_line(w_iv, "Linear Programming Train")
plot_line(w_iv, "Linear Programming Test", train=False)
calculate_Ein_Etest(X,X_test,w_iv, "Linear Programming")




# 3rd Orders:
def third_order_features(symmetries, intensities):
    X_third_order = []
    
    for s, i in zip(symmetries, intensities):
        features = [
            1,                  # all_1 (bias term)
            s,                  # symmetries (x1)
            i,                  # intensities (x2)
            s**2,               # symmetries_squared (x1^2)
            s * i,              # symmetries_intensities (x1 * x2)
            i**2,               # intensities_squared (x2^2)
            s**3,               # symmetries_cubed (x1^3)
            (s**2) * i,         # symmetries_squared_intensities (x1^2 * x2)
            s * (i**2),         # symmetries_intensities_squared (x1 * x2^2)
            i**3                # intensities_cubed (x2^3)
        ]
        X_third_order.append(features)
    
    return np.array(X_third_order)
def plot_decision_boundary_3rd_order(weights, title, train=True):
    if train:
        for i in range(len(symmetries)):
            color = 'blue' if labels[i] == 'o' else 'red'
            plt.scatter(symmetries[i], intensities[i], marker=labels[i], c=color)
    else:
        for i in range(len(symmetries_t)):
            color = 'blue' if labels_t[i] == 'o' else 'red'
            plt.scatter(symmetries_t[i], intensities_t[i], marker=labels_t[i], c=color)
    
    x_min, x_max = min(symmetries) - 0.1, max(symmetries) + 0.1
    y_min, y_max = min(intensities) - 0.1, max(intensities) + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    grid_features = third_order_features(xx.ravel(), yy.ravel())
    Z = np.dot(grid_features, weights)
    Z = Z.reshape(xx.shape) 
    
    plt.contour(xx, yy, Z, levels=[0], colors='green', linewidths=2)
    plt.ylim(y_min, y_max)
    plt.xlim(x_min, x_max)
    plt.title(f'{title}')
    plt.xlabel('Symmetry')
    plt.ylabel('Average Intensity')
    plt.legend(["Decision Boundary"])
    plt.show()
X_third_order = third_order_features(symmetries, intensities)
X_third_order_test = third_order_features(symmetries_t, intensities_t)
starting_third_order_weights = np.zeros(X_third_order.shape[1])

# (i) Linear Regression for classification followed by pocket for improvement.
def third_order_linear_regression_classification(X, y):
    weights = np.linalg.pinv(X.T @ X) @ X.T @ y
    return weights

    # Pocket Algorithm for improvement
def third_order_pocket_algorithm(X, y, weights, max_iter=10000):
    best_weights = weights
    best_accuracy = accuracy_score(y, np.sign(X @ weights))
    
    for _ in range(max_iter):
        predictions = np.sign(X @ weights)
        misclassified = np.where(predictions != y)[0]
        
        if len(misclassified) == 0:
            break

        i = np.random.choice(misclassified)
        weights += y[i] * X[i]
        
        accuracy = accuracy_score(y, np.sign(X @ weights))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_weights = weights.copy()
    
    return best_weights
w_i = starting_third_order_weights.copy()
w_i = third_order_linear_regression_classification(X_third_order,y)
w_i = third_order_pocket_algorithm(X_third_order,y,w_i)
plot_decision_boundary_3rd_order(w_i, "3rd Linear Regression with Pocket Train")
plot_decision_boundary_3rd_order(w_i, "3rd Linear Regression with Pocket Test", train=False)
calculate_Ein_Etest(X_third_order, X_third_order_test, w_i, "Linear Regression with Pocket")

# (ii) Logistic regression for classification using gradient descent.
def third_order_gradient_descent(X, y, weights, learning_rate=0.1, max_iter=100000):
    N = len(X)  # Number of samples
    weights = weights.astype(float)  # Ensure weights are float

    for _ in range(max_iter):
        vt = np.zeros_like(weights, dtype=float)  # Initialize vt as float
        for i in range(N):
            xi = X[i]
            yi = y[i]
            
            # Calculate vt as the gradient for this example
            vt += (yi * xi) / (1 + math.pow(math.e, yi * (np.dot(weights, xi))))
        
        # Update weights with the average gradient
        weights += learning_rate * vt / N

    return weights
w_ii = starting_third_order_weights.copy()
w_ii = third_order_gradient_descent(X_third_order, y, w_ii)
plot_decision_boundary_3rd_order(w_ii, "3rd Gradient Descent Train")
plot_decision_boundary_3rd_order(w_ii, "3rd Gradient Descent Test", train=False)
calculate_Ein_Etest(X_third_order,X_third_order_test,w_ii, "Gradient Descent")

# (iii) Logistic regression for classification using stochastic gradient descent.
def third_order_stochastic_gradient_descent(X, y, weights, learning_rate=0.1, max_iter=100000):
    N = len(X)  # Number of samples
    weights = weights.astype(float)  # Ensure weights are float

    for _ in range(max_iter):
        for i in range(N):
            xi = X[i]
            yi = y[i]
            
            # Calculate the gradient for the current example
            gradient = (yi * xi) / (1 + np.exp(yi * np.dot(xi, weights)))
            
            # Update weights based on the gradient of the current sample
            weights += learning_rate * gradient

    return weights
w_iii = starting_third_order_weights.copy()
w_iii = third_order_stochastic_gradient_descent(X_third_order, y, w_iii)
plot_decision_boundary_3rd_order(w_iii, "3rd Stochastic Gradient Descent Train")
plot_decision_boundary_3rd_order(w_iii, "3rd Stochastic Gradient Descent Test", train=False)
calculate_Ein_Etest(X_third_order,X_third_order_test,w_iii, "Stochastic Gradient Descent")

# (iv) Linear Programming for classification (Graduate, 6xxx-level, only).
def third_order_linear_programming_classification(X, y, learning_rate=0.1, max_iter=100000):
    # Initialize weights as small random values
    weights = np.random.randn(X.shape[1]) * learning_rate
    
    for _ in range(max_iter):
        for i in range(len(y)):
            xi = X[i]
            yi = y[i]
            constraint = yi * np.dot(weights, xi) # y*(w*x) >= 1
            
            # If constraint is violated
            if constraint < 1:
                weights += learning_rate * yi * xi
            else:
                weights -= learning_rate * weights * 0.01  # Small decay to reduce norm

    return weights
w_iv = third_order_linear_programming_classification(X_third_order, y)
plot_decision_boundary_3rd_order(w_iv, "3rd Linear Programming Train")
plot_decision_boundary_3rd_order(w_iv, "3rd Linear Programming Test", train=False)
calculate_Ein_Etest(X_third_order,X_third_order_test,w_iv, "Linear Programming")