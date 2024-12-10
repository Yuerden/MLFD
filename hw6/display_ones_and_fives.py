import numpy as np
import matplotlib.pyplot as plt

# Initialize lists to hold labels and images
labels = []
images = []

# Open and read the zip.train file
with open('zip.train', 'r') as file:
    for line in file:
        # Split the line into components
        parts = line.strip().split()
        
        # Extract the label and pixel values
        label = float(parts[0])
        pixel_values = np.array(parts[1:], dtype=float)
        
        # Only process digits '1' and '5'
        if label == 1.0 or label == 5.0:
            labels.append(int(label))
            images.append(pixel_values)

# Convert lists to NumPy arrays for easier indexing
labels = np.array(labels)
images = np.array(images)

# Find indices of digits '1' and '5'
indices_digit1 = np.where(labels == 1)[0]
indices_digit5 = np.where(labels == 5)[0]

# Select one example of each digit
index1 = indices_digit1[0]  # First instance of digit '1'
index5 = indices_digit5[0]  # First instance of digit '5'

# Image dimensions
img_size = 16  # Since images are 16x16 pixels

# Reshape and display digit '1'
image1 = images[index1].reshape((img_size, img_size))
plt.figure()
plt.imshow(image1, cmap='gray')
plt.title('Digit 1')
plt.axis('off')  # Hide axis
plt.show()

# Reshape and display digit '5'
image5 = images[index5].reshape((img_size, img_size))
plt.figure()
plt.imshow(image5, cmap='gray')
plt.title('Digit 5')
plt.axis('off')  # Hide axis
plt.show()
