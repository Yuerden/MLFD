import math
import numpy as np
import matplotlib.pyplot as plt

# Define function and its partial derivatives
def f(x, y):
    return x * x + 2 * y * y + 2 * math.sin(2 * math.pi * x) * math.sin(2 * math.pi * y)

def simple_x(x, y):
    return 2 * x + 4 * math.pi * math.sin(2 * math.pi * y) * math.cos(2 * math.pi * x)

def simple_y(x, y):
    return 4 * y + 4 * math.pi * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)

# Gradient descent function
def gradient_descent(x_start, y_start, learning_rate, iterations=50):
    x, y = x_start, y_start
    values = [f(x, y)]
    iter_counts = [0]

    for i in range(1, iterations + 1):
        grad_x = simple_x(x, y)
        grad_y = simple_y(x, y)

        x -= learning_rate * grad_x
        y -= learning_rate * grad_y

        values.append(f(x, y))
        iter_counts.append(i)

    print(f"Start (x, y): ({x_start}, {y_start})")
    print(f"Minimum (x, y): ({x:.4f}, {y:.4f})")
    print(f"Minimum value: {values[-1]:.4f}")
    
    return iter_counts, values, x, y

# Part (a) - Gradient descent with different learning rates
learning_rates = [0.01, 0.1]
initial_point = (0.1, 0.1)
iterations = 50

plt.figure(figsize=(10, 5))
for eta in learning_rates:
    iter_counts, values, _, _ = gradient_descent(initial_point[0], initial_point[1], eta, iterations)
    plt.plot(iter_counts, values, label=f"Learning rate = {eta}")

plt.xlabel("Iterations")
plt.ylabel("f(x, y)")
plt.title("Function value drop over iterations for different learning rates")
plt.legend()
plt.show()

# Part (b) - Gradient descent from different starting points with eta = 0.01
learning_rate = 0.01
starting_points = [(0.1, 0.1), (1, 1), (-0.5, -0.5), (-1, -1)]

print("\nStarting point | Final (x, y)   | Minimum value")
print("---------------|----------------|---------------")
for start in starting_points:
    iter_counts, values, x_min, y_min = gradient_descent(start[0], start[1], learning_rate, iterations)
    min_value = values[-1]
    print(f"{start}       | ({x_min:.4f}, {y_min:.4f}) | {min_value:.4f}\n")
