import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 6  # number of coin flips
mu = 0.5  # probability of heads
epsilon_values = np.linspace(0, 1, 100)  # epsilon range
num_coins = 2  # number of coins

# Function to calculate exact probability using binomial distribution
def binomial_probability(N, mu, epsilon):
    k_values = np.arange(0, N + 1)
    p_k = np.array([np.math.comb(N, k) * mu**k * (1 - mu)**(N - k) for k in k_values])
    nu_k = k_values / N
    p_out_of_bound = np.sum(p_k[np.abs(nu_k - mu) > epsilon])
    return p_out_of_bound

# Calculate the exact probability P[max_i |nu_i - mu_i| > epsilon] for two coins
prob_max_eps = []
for epsilon in epsilon_values:
    p_single_coin = binomial_probability(N, mu, epsilon)
    p_max_eps = 2 * p_single_coin - p_single_coin**2  # P[A] + P[B] - P[A]P[B]
    prob_max_eps.append(p_max_eps)

# Calculate Hoeffding bound for comparison
hoeffding_bound = num_coins * 2 * np.exp(-2 * N * epsilon_values**2)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, prob_max_eps, label=r'$P[\max_i |\nu_i - \mu_i| > \epsilon]$', color='blue')
plt.plot(epsilon_values, hoeffding_bound, label='Hoeffding Bound', color='red', linestyle='--')
plt.title(r'Probability $P[\max_i |\nu_i - \mu_i| > \epsilon]$ vs Hoeffding Bound')
plt.xlabel(r'$\epsilon$')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
