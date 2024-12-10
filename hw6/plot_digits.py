import matplotlib.pyplot as plt

# Open the training file
with open("zip.train", "r") as train:
    labels = []
    symmetries = []
    intensities = []

    for line in train:
        # Split the line into label and pixel values
        data = line.strip().split()
        label = data[0]

        # Process only digits '1' and '5'
        if label == "1.0000" or label == "5.0000":
            # Convert pixel values to floats
            values = list(map(float, data[1:]))

            # Append the label
            if label == "1.0000":
                labels.append('o')
            else:  # label == "5.0000"
                labels.append('x')

            # Symmetry calculation
            cur_sym = 0
            for i in range(16):  # For each row
                for j in range(8):  # For half the columns
                    left_pixel = values[i*16 + j]
                    right_pixel = values[i*16 + (15 - j)]
                    cur_sym += abs(left_pixel - right_pixel)
            symmetries.append(cur_sym / 256)

            # Intensity calculation
            cur_intense = sum(values)
            intensities.append(cur_intense / 256)

# Plotting
for i in range(len(symmetries)):
    color = 'blue' if labels[i] == 'o' else 'red'
    plt.scatter(symmetries[i], intensities[i], marker=labels[i], c=color)

plt.title('2-D Scatter Plot of Features')
plt.xlabel('Symmetry')
plt.ylabel('Average Intensity')
plt.show()
