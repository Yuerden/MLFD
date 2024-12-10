import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Define the data points
X = np.array([[1, 0],
              [0, 1],
              [0, -1],
              [-1, 0],
              [0, 2],
              [0, -2],
              [-2, 0]])
y = np.array([-1, -1, -1, -1, +1, +1, +1])

# Part (a): 1-NN and 3-NN in x-space
# Train the classifiers
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X, y)

knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X, y)

# Create a meshgrid for plotting decision boundaries
x1_min, x1_max = -3, 3
x2_min, x2_max = -3, 3
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 500),
                       np.linspace(x2_min, x2_max, 500))
grid_points = np.c_[xx1.ravel(), xx2.ravel()]

# Predict labels for each point in the grid
Z1 = knn1.predict(grid_points).reshape(xx1.shape)
Z3 = knn3.predict(grid_points).reshape(xx1.shape)

# Plot the decision boundaries for 1-NN and 3-NN
plt.figure(figsize=(12, 5))

# 1-NN Decision Boundary
plt.subplot(1, 2, 1)
plt.contourf(xx1, xx2, Z1, alpha=0.4, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
plt.title('Part (a) - 1-NN Decision Boundary')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

# 3-NN Decision Boundary
plt.subplot(1, 2, 2)
plt.contourf(xx1, xx2, Z3, alpha=0.4, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
plt.title('Part (a) - 3-NN Decision Boundary')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

plt.tight_layout()
plt.show()

# Part (b): Non-linear transformation and classification in z-space
def x_to_z(x):
    r = np.sqrt(x[:, 0]**2 + x[:, 1]**2)
    theta = np.arctan2(x[:, 1], x[:, 0])
    return np.column_stack((r, theta))

# Transform the data to z-space
Z = x_to_z(X)

# Train the classifiers in z-space
knn1_z = KNeighborsClassifier(n_neighbors=1)
knn1_z.fit(Z, y)

knn3_z = KNeighborsClassifier(n_neighbors=3)
knn3_z.fit(Z, y)

# Transform the grid points to z-space
grid_points_z = x_to_z(grid_points)

# Predict labels for each point in the grid (in z-space)
Z1_b = knn1_z.predict(grid_points_z).reshape(xx1.shape)
Z3_b = knn3_z.predict(grid_points_z).reshape(xx1.shape)

# Plot the decision boundaries after transformation
plt.figure(figsize=(12, 5))

# 1-NN Decision Boundary after transformation
plt.subplot(1, 2, 1)
plt.contourf(xx1, xx2, Z1_b, alpha=0.4, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
plt.title('Part (b) - 1-NN Decision Boundary after Transformation')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

# 3-NN Decision Boundary after transformation
plt.subplot(1, 2, 2)
plt.contourf(xx1, xx2, Z3_b, alpha=0.4, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.coolwarm)
plt.title('Part (b) - 3-NN Decision Boundary after Transformation')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)

plt.tight_layout()
plt.show()
