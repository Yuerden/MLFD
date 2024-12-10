import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Parameters
thk = 5
rad = 10
sep = 5       # separation between the semi-circles
inner = rad  # inner radius
outer = rad + thk  # outer radius
num_points = 2000  # total number of points

# Generate semi-circle function
def generate_semi(rad, thk, sep, num_points, label):
    origin_x = thk + rad
    origin_y = thk + rad + sep
    theta = np.pi * np.random.rand(num_points)  # angles between 0 and pi
    if label == -1:  # bottom semi-circle
        origin_x += thk / 2 + rad
        origin_y -= sep
        theta = np.pi + np.pi * np.random.rand(num_points)
    r = (outer - inner) * np.random.rand(num_points) + inner  # radii between inner and outer
    x = origin_x + r * np.cos(theta)
    y = origin_y + r * np.sin(theta)
    labels = np.full(num_points, label)
    return x, y, labels

# Generate semi-circles
x_top, y_top, labels_top = generate_semi(rad, thk, sep, num_points // 2, +1)
x_bottom, y_bottom, labels_bottom = generate_semi(rad, thk, sep, num_points // 2, -1)

# Combine the data for both semi-circles
x_coords = np.concatenate([x_top, x_bottom])
y_coords = np.concatenate([y_top, y_bottom])
labels = np.concatenate([labels_top, labels_bottom])

# Prepare data for KNN
X_data = np.column_stack((x_coords, y_coords))
y_labels = labels  # labels: +1 or -1

# Create meshgrid for plotting decision boundaries
x_min, x_max = x_coords.min() - 1, x_coords.max() + 1
y_min, y_max = y_coords.min() - 1, y_coords.max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),
                     np.linspace(y_min, y_max, 500))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# Train 1-NN classifier
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_data, y_labels)

# Predict labels for the grid
Z1 = knn1.predict(grid_points)
Z1 = Z1.reshape(xx.shape)

# Train 3-NN classifier
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_data, y_labels)

# Predict labels for the grid
Z3 = knn3.predict(grid_points)
Z3 = Z3.reshape(xx.shape)

# Plotting decision regions
plt.figure(figsize=(14, 6))

# 1-NN decision boundary
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z1, alpha=0.4, cmap=plt.cm.coolwarm)
plt.scatter(x_top, y_top, c='blue', label='+1', edgecolor='k', alpha=0.5)
plt.scatter(x_bottom, y_bottom, c='red', label='-1', edgecolor='k', alpha=0.5)
plt.title('1-NN Decision Boundary')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 3-NN decision boundary
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z3, alpha=0.4, cmap=plt.cm.coolwarm)
plt.scatter(x_top, y_top, c='blue', label='+1', edgecolor='k', alpha=0.5)
plt.scatter(x_bottom, y_bottom, c='red', label='-1', edgecolor='k', alpha=0.5)
plt.title('3-NN Decision Boundary')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()
