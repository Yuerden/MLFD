import numpy as np
import matplotlib.pyplot as plt

# Define the perceptron function h(x) = sign(w^T x)
def perceptron_decision_boundary(w, x1_range):
    # w = [w_0, w_1, w_2], x2 = -(w_1/w_2) * x1 - (w_0/w_2)
    a = -w[1] / w[2]  # slope
    b = -w[0] / w[2]  # intercept
    return a * x1_range + b

def plot_decision_boundary(x2, x1_range, w, color_positive, color_negative, title):
    # Plot the decision boundaries
    plt.figure(figsize=(8, 6))

    # Plot the decision boundary
    plt.plot(x1_range, x2, label=f"Decision boundary for {w}", color="black")

    # Fill the region above the line (h(x) = +1) with the specified color for positive
    plt.fill_between(x1_range, x2, 5, where=(x1_range > -5), color=color_positive, alpha=0.5, label="h(x) = -1 region")

    # Fill the region below the line (h(x) = -1) with the specified color for negative
    plt.fill_between(x1_range, x2, -5, where=(x1_range > -5), color=color_negative, alpha=0.5, label="h(x) = +1 region")

    # Adding labels and a title
    plt.xlabel(r'$x_1$', fontsize=12)
    plt.ylabel(r'$x_2$', fontsize=12)
    plt.title(title)

    # Add grid, legend, and show the plot
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()

# Generate x1 values
x1_range = np.linspace(-5, 5, 100)

# Define weights w = [1, 2, 3]^T
w_positive = np.array([1, 2, 3])

# Define weights w = -[1, 2, 3]^T
w_negative = np.array([-1, -2, -3])

# Calculate corresponding x2 values for the decision boundary
x2_positive = perceptron_decision_boundary(w_positive, x1_range)
x2_negative = perceptron_decision_boundary(w_negative, x1_range)

# Plot the results for w = [1, 2, 3]^T
plot_decision_boundary(x2_positive, x1_range, w_positive, color_positive='blue', color_negative='red',
                       title='Perceptron Decision Boundary for w = [1, 2, 3]^T')

# Plot the results for w = [-1, -2, -3]^T
plot_decision_boundary(x2_negative, x1_range, w_negative, color_positive='red', color_negative='blue',
                       title='Perceptron Decision Boundary for w = -[1, 2, 3]^T')
