import numpy as np
import matplotlib.pyplot as plt

plt.plot(1, 0, '.')
plt.plot(-1, 0, '.')
plt.axvline(x = 0)

# Points
x = np.linspace(-1, 1, 100)
y = x**3

# Plot Points
plt.plot(x, y)

plt.title('Decision Boundaries in X-Space')
plt.show()
