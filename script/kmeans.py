from tslearn.utils import save_time_series_txt, load_time_series_txt
X = load_time_series_txt("newTest.txt")
print(X.shape)
from tslearn.clustering import TimeSeriesKMeans
km = TimeSeriesKMeans(n_clusters=3, metric="dtw")
labels = km.fit_predict(X)
print(km.cluster_centers_)
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Assuming you have the waveforms in an array called 'waveforms' with shape (1000, 46)
waveforms = X

# Creating a meshgrid for time and waveform index
time_points = np.arange(waveforms.shape[1])
waveform_indices = np.arange(waveforms.shape[0])
time_mesh, waveform_mesh = np.meshgrid(time_points, waveform_indices)

# Plotting the waveforms as a heatmap
plt.pcolormesh(time_mesh, waveform_mesh, waveforms, cmap='viridis')

# Customize the plot
plt.colorbar(label='Amplitude')
plt.xlabel('Time')
plt.ylabel('Waveform Index')
plt.title('Waveform Heatmap')

# Show the plot
plt.show()

