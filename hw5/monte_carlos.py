import numpy as np
import matplotlib.pyplot as plt

S = 10000  # # Simulations (# of 2 points and g_i(x)'s )
T = 1000   # # Test points
x_test = np.linspace(-1, 1, T)  # Vector of test points

a_vals = np.zeros(S)
b_vals = np.zeros(S)
g_bar_x = np.zeros(T)
E_out_list = np.zeros(S)

f_x = x_test**2

for i in range(S):
    # Generate training points
    x1, x2 = np.random.uniform(-1, 1, 2)
    y1, y2 = x1**2, x2**2 # calculate y values

    # Calculate g_i_x coefficients
    if x1 != x2:
        a_i = (y2 - y1) / (x2 - x1)
    else:
        a_i = 0
    b_i = y1 - a_i * x1

    # Store parameters for g_i_x and Var_x calculations later
    a_vals[i] = a_i
    b_vals[i] = b_i

    g_i_x = a_i * x_test + b_i
    g_bar_x += g_i_x / S

    E_out_i = np.mean((g_i_x - f_x)**2)
    E_out_list[i] = E_out_i

# Calculate Results
E_out = np.mean(E_out_list)
Bias = np.mean((g_bar_x - f_x)**2)
Var_x = np.zeros(T)
for i in range(S):
    g_i_x = a_vals[i] * x_test + b_vals[i]
    Var_x += ((g_i_x - g_bar_x)**2) / S
Variance = np.mean(Var_x)

# Print
print(f"E_out: {E_out:.4f}")
print(f"Bias: {Bias:.4f}")
print(f"Variance: {Variance:.4f}")
print(f"Bias + Variance: {Bias + Variance:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(x_test, f_x, label='Target Function f(x) = x^2', color='blue')
plt.plot(x_test, g_bar_x, label='Average Hypothesis ḡ(x)', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Average Hypothesis vs. Target Function')
plt.legend()
plt.grid(True)
plt.show()
