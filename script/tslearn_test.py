from tslearn.utils import save_time_series_txt, load_time_series_txt
X = load_time_series_txt("newTest.txt")
print(X.shape)
from tslearn.clustering import TimeSeriesKMeans
#km = TimeSeriesKMeans(n_clusters=5, metric="dtw")
km = TimeSeriesKMeans(n_clusters=5, metric="softdtw")
labels = km.fit_predict(X)
print(km.cluster_centers_)
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have the waveforms in an array called 'waveforms' with shape (1000, 46)
waveforms = X

# Assuming you have the center waveforms in an array called 'center_waveforms' with shape (2, 46)
center_waveforms = km.cluster_centers_

# Plotting the waveforms
for waveform in waveforms:
    plt.plot(waveform, color='blue', alpha=0.05)

# Plotting the center waveforms
for center_waveform in center_waveforms:
    plt.plot(center_waveform, linewidth=2)

# Customize the plot
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Waveforms with Cluster Centers')

# Show the plot
plt.show()

