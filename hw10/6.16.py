import numpy as np
import matplotlib.pyplot as plt
import random
import math
import time

# Problem 6.16

# Set the random seed for reproducibility
random.seed(1)
np.random.seed(1)

num_points = 10000

# Part (a): Uniformly Distributed Data in [0,1]^2

# Generate data points uniformly in [0,1]^2
dataX = [random.uniform(0, 1) for _ in range(num_points)]
dataY = [random.uniform(0, 1) for _ in range(num_points)]

# Simple Greedy Heuristic for 10-partition
# Initialize the first center randomly
center_indices = [random.randint(0, num_points - 1)]
for _ in range(9):  # We need a total of 10 centers
    max_distance = -1
    new_center = -1
    for i in range(num_points):
        if i in center_indices:
            continue
        # Find the minimum distance to the existing centers
        min_dist_to_centers = min(math.dist([dataX[i], dataY[i]], [dataX[c], dataY[c]]) for c in center_indices)
        if min_dist_to_centers > max_distance:
            max_distance = min_dist_to_centers
            new_center = i
    center_indices.append(new_center)

# Partition the data based on the nearest center
partitions = [[] for _ in range(10)]
colors = []
color_palette = ['steelblue', 'seagreen', 'skyblue', 'mediumpurple', 'salmon',
                 'orange', 'gold', 'thistle', 'aquamarine', 'pink']
for i in range(num_points):
    # Find the nearest center
    distances_to_centers = [math.dist([dataX[i], dataY[i]], [dataX[c], dataY[c]]) for c in center_indices]
    nearest_center = np.argmin(distances_to_centers)
    partitions[nearest_center].append(i)
    colors.append(color_palette[nearest_center])

# Calculate the radius for each partition
radii = []
for idx, center in enumerate(center_indices):
    max_radius = max(math.dist([dataX[i], dataY[i]], [dataX[center], dataY[center]]) for i in partitions[idx])
    radii.append(max_radius)

# Plot the data points and color code them based on partitions
plt.figure(figsize=(8, 8))
plt.scatter(dataX, dataY, c=colors, s=5)
for idx, center in enumerate(center_indices):
    plt.scatter(dataX[center], dataY[center], c='red', marker='x', s=100)
plt.title('Data Points Partitioned into 10 Clusters (Uniform Data)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Generate 10,000 random query points uniformly in [0,1]^2
testDataX = [random.uniform(0, 1) for _ in range(num_points)]
testDataY = [random.uniform(0, 1) for _ in range(num_points)]

# Brute Force Nearest Neighbor Search
start_time = time.time()
for i in range(num_points):
    min_dist = float('inf')
    nn = -1
    for j in range(num_points):
        dist = math.dist([testDataX[i], testDataY[i]], [dataX[j], dataY[j]])
        if dist < min_dist:
            min_dist = dist
            nn = j
brute_force_time = time.time() - start_time
print(f"Brute Force Time (Uniform Data): {brute_force_time:.2f} seconds")

# Branch and Bound Nearest Neighbor Search
start_time = time.time()
for i in range(num_points):
    # Find the nearest center
    distances_to_centers = [math.dist([testDataX[i], testDataY[i]], [dataX[c], dataY[c]]) for c in center_indices]
    nearest_center = np.argmin(distances_to_centers)
    min_dist = float('inf')
    nn = -1
    # Search within the nearest cluster
    for idx in partitions[nearest_center]:
        dist = math.dist([testDataX[i], testDataY[i]], [dataX[idx], dataY[idx]])
        if dist < min_dist:
            min_dist = dist
            nn = idx
    # Check if we need to search other clusters
    threshold = min_dist
    for k, center in enumerate(center_indices):
        if k != nearest_center:
            if distances_to_centers[k] - radii[k] < threshold:
                # Need to search this cluster as well
                for idx in partitions[k]:
                    dist = math.dist([testDataX[i], testDataY[i]], [dataX[idx], dataY[idx]])
                    if dist < min_dist:
                        min_dist = dist
                        nn = idx
branch_and_bound_time = time.time() - start_time
print(f"Branch and Bound Time (Uniform Data): {branch_and_bound_time:.2f} seconds")

# Part (b): Data from a Mixture of 10 Gaussians

# Generate 10 Gaussian centers randomly in [0,1]^2
gaussian_centers = np.random.uniform(0, 1, size=(10, 2))
sigma = 0.1
covariance = sigma ** 2 * np.eye(2)

# Generate data points from the mixture of Gaussians
dataX = []
dataY = []
for i in range(num_points):
    gaussian_idx = random.randint(0, 9)
    point = np.random.multivariate_normal(gaussian_centers[gaussian_idx], covariance)
    dataX.append(point[0])
    dataY.append(point[1])

# Use the Gaussian centers as cluster centers
center_indices = []
for center in gaussian_centers:
    # Find the closest point in data to the center
    min_dist = float('inf')
    center_idx = -1
    for i in range(num_points):
        dist = math.dist([dataX[i], dataY[i]], center)
        if dist < min_dist:
            min_dist = dist
            center_idx = i
    center_indices.append(center_idx)

# Partition the data based on the nearest center
partitions = [[] for _ in range(10)]
colors = []
for i in range(num_points):
    # Find the nearest center
    distances_to_centers = [math.dist([dataX[i], dataY[i]], [dataX[c], dataY[c]]) for c in center_indices]
    nearest_center = np.argmin(distances_to_centers)
    partitions[nearest_center].append(i)
    colors.append(color_palette[nearest_center])

# Calculate the radius for each partition
radii = []
for idx, center in enumerate(center_indices):
    max_radius = max(math.dist([dataX[i], dataY[i]], [dataX[center], dataY[center]]) for i in partitions[idx])
    radii.append(max_radius)

# Plot the data points and color code them based on partitions
plt.figure(figsize=(8, 8))
plt.scatter(dataX, dataY, c=colors, s=5)
for idx, center in enumerate(center_indices):
    plt.scatter(dataX[center], dataY[center], c='red', marker='x', s=100)
plt.title('Data Points Partitioned into 10 Clusters (Gaussian Mixture Data)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Generate 10,000 random query points from the same mixture of Gaussians
testDataX = []
testDataY = []
for i in range(num_points):
    gaussian_idx = random.randint(0, 9)
    point = np.random.multivariate_normal(gaussian_centers[gaussian_idx], covariance)
    testDataX.append(point[0])
    testDataY.append(point[1])

# Brute Force Nearest Neighbor Search
start_time = time.time()
for i in range(num_points):
    min_dist = float('inf')
    nn = -1
    for j in range(num_points):
        dist = math.dist([testDataX[i], testDataY[i]], [dataX[j], dataY[j]])
        if dist < min_dist:
            min_dist = dist
            nn = j
brute_force_time = time.time() - start_time
print(f"Brute Force Time (Gaussian Mixture Data): {brute_force_time:.2f} seconds")

# Branch and Bound Nearest Neighbor Search
start_time = time.time()
for i in range(num_points):
    # Find the nearest center
    distances_to_centers = [math.dist([testDataX[i], testDataY[i]], [dataX[c], dataY[c]]) for c in center_indices]
    nearest_center = np.argmin(distances_to_centers)
    min_dist = float('inf')
    nn = -1
    # Search within the nearest cluster
    for idx in partitions[nearest_center]:
        dist = math.dist([testDataX[i], testDataY[i]], [dataX[idx], dataY[idx]])
        if dist < min_dist:
            min_dist = dist
            nn = idx
    # Check if we need to search other clusters
    threshold = min_dist
    for k, center in enumerate(center_indices):
        if k != nearest_center:
            if distances_to_centers[k] - radii[k] < threshold:
                # Need to search this cluster as well
                for idx in partitions[k]:
                    dist = math.dist([testDataX[i], testDataY[i]], [dataX[idx], dataY[idx]])
                    if dist < min_dist:
                        min_dist = dist
                        nn = idx
branch_and_bound_time = time.time() - start_time
print(f"Branch and Bound Time (Gaussian Mixture Data): {branch_and_bound_time:.2f} seconds")
