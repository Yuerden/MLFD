import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# Parameters
N = 6          # Number of tosses per coin
mu = 0.5       # Probability of heads for both coins
num_coins = 2  # Number of coins

# Possible values of k (number of heads) and corresponding nu values
k_values = np.arange(N + 1)
nu_values = k_values / N

# Binomial probabilities for each k
P_k = comb(N, k_values) * (mu ** k_values) * ((1 - mu) ** (N - k_values))

# Absolute deviations |nu - mu| for each possible outcome
deviations = np.abs(nu_values - mu)

# Generate a fine grid of epsilon values between 0 and 0.5
epsilon_values = np.linspace(0, 0.5, 1000)
P_single = np.zeros_like(epsilon_values)

# Calculate P_single for each epsilon
for i, epsilon in enumerate(epsilon_values):
    indices = deviations > epsilon
    P_single[i] = np.sum(P_k[indices])

# Calculate P_max for two coins
P_max = 2 * P_single - P_single ** 2

# Calculate the Hoeffding bound for each epsilon
Hoeffding_bound = 4 * np.exp(-2 * N * epsilon_values ** 2)

# Ensure the Hoeffding bound does not exceed 1
Hoeffding_bound = np.minimum(Hoeffding_bound, 1)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, P_max, label='Exact Probability', color='blue')
plt.plot(epsilon_values, Hoeffding_bound, label='Hoeffding Bound', linestyle='--', color='red')
plt.xlabel(r'$\epsilon$', fontsize=14)
plt.ylabel(r'$P\left[\max_i |\nu_i - \mu| > \epsilon\right]$', fontsize=14)
plt.title('Probability vs. Hoeffding Bound for N=6, μ=0.5, 2 Coins', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
