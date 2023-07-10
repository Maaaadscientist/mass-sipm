import matplotlib.pyplot as plt
import numpy as np

mean_list = [1.0, 2.5, 4.0, 5.5]
mean_error_list = [0.1, 0.2, 0.3, 0.4]

# Calculate peak distances
peak_distances = np.diff(mean_list)

# Calculate uncertainties
uncertainties = np.sqrt(np.diff(mean_error_list)**2 + np.diff(mean_error_list)**2)

# Generate x-axis labels
x_labels = [f"Peak{i}-{i+1}" for i in range(len(peak_distances))]

# Plotting
x_values = np.arange(len(peak_distances))
plt.errorbar(x_values, peak_distances, yerr=uncertainties, fmt='o', capsize=5)
plt.xticks(x_values, x_labels)  # Set x-axis labels
plt.xlabel('Peak Pair')
plt.ylabel('Distance')
plt.title('Peak Distances with Uncertainties')
plt.show()

