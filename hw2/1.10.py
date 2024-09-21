import numpy as np
import matplotlib.pyplot as plt

# Number of simulations
num_simulations = 100000        #1 or 100000 times, depending on which part
num_coins = 1000
num_flips = 10

# Initialize lists to store fractions of heads
nu_1_list = []
nu_rand_list = []
nu_min_list = []

# Perform the experiment
for _ in range(num_simulations):
    # Simulate flipping 1000 coins 10 times
    flips = np.random.binomial(1, 0.5, (num_coins, num_flips))
    
    # Calculate the fraction of heads for each coin
    fractions_of_heads = np.mean(flips, axis=1)
    
    # c_1: The first coin
    nu_1 = fractions_of_heads[0]
    
    # c_rand: A randomly chosen coin
    nu_rand = np.random.choice(fractions_of_heads)
    
    # c_min: The coin with the minimum number of heads
    nu_min = np.min(fractions_of_heads)
    
    # Store the results
    nu_1_list.append(nu_1)
    nu_rand_list.append(nu_rand)
    nu_min_list.append(nu_min)

# Convert to numpy arrays for easy manipulation
nu_1_list = np.array(nu_1_list)
nu_rand_list = np.array(nu_rand_list)
nu_min_list = np.array(nu_min_list)

# Plot histograms of nu_1, nu_rand, and nu_min
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(nu_1_list, bins=20, range=(0, 1), alpha=0.7, label='nu_1')
plt.title("Distribution of nu_1 (First Coin)")
plt.xlabel("Fraction of Heads")
plt.ylabel("Frequency")
plt.xticks(np.arange(0, 1.1, 0.1))

plt.subplot(1, 3, 2)
plt.hist(nu_rand_list, bins=20, range=(0, 1), alpha=0.7, label='nu_rand')
plt.title("Distribution of nu_rand (Random Coin)")
plt.xlabel("Fraction of Heads")
plt.ylabel("Frequency")
plt.xticks(np.arange(0, 1.1, 0.1))

plt.subplot(1, 3, 3)
plt.hist(nu_min_list, bins=20, range=(0, 1), alpha=0.7, label='nu_min')
plt.title("Distribution of nu_min (Coin with Min Heads)")
plt.xlabel("Fraction of Heads")
plt.ylabel("Frequency")
plt.xticks(np.arange(0, 1.1, 0.1))

plt.tight_layout()
plt.show()

# Part (c): Estimating P[|nu - mu| > epsilon] and comparing with Hoeffding Bound
epsilon_values = np.linspace(0, 0.5, 100)
mu = 0.5

# Compute P[|nu - mu| > epsilon] for each coin
def empirical_prob(nu_list, epsilon_values, mu=0.5):
    probs = []
    for epsilon in epsilon_values:
        prob = np.mean(np.abs(nu_list - mu) > epsilon)
        probs.append(prob)
    return np.array(probs)

P_nu_1 = empirical_prob(nu_1_list, epsilon_values)
P_nu_rand = empirical_prob(nu_rand_list, epsilon_values)
P_nu_min = empirical_prob(nu_min_list, epsilon_values)

# Compute the Hoeffding bound
N = num_flips
hoeffding_bound = 2 * np.exp(-2 * (epsilon_values**2) * N)
# hoeffding_bound = num_simulations * 2 * np.exp(-2 * (epsilon_values**2) * N)  # Multiple bins

# Plot the probabilities and Hoeffding bound
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, P_nu_1, label='P[|nu_1 - mu| > epsilon]', linestyle='--')
plt.plot(epsilon_values, P_nu_rand, label='P[|nu_rand - mu| > epsilon]', linestyle='-.')
plt.plot(epsilon_values, P_nu_min, label='P[|nu_min - mu| > epsilon]', linestyle=':')
plt.plot(epsilon_values, hoeffding_bound, label='Hoeffding Bound', color='black', linewidth=2)

plt.title('P[|nu - mu| > epsilon] vs. Hoeffding Bound')
plt.xlabel('epsilon')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()
