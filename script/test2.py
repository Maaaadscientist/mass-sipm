import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv("fit_results.csv")

# Calculate peak distances
df['peak_distance_1_2'] = df.groupby(['run_number', 'channel', 'type', 'position', 'run_type', 'voltage'])['mean']
df['peak_distance_2_3'] = df.groupby(['run_number', 'channel', 'type', 'position', 'run_type', 'voltage'])['mean']

run_number = 109 
channel = 2 
type_ = 'tile' 
position = 6 
run_type = 'main' 
voltage = 2 
peak = 0 
print(df['peak_distance_1_2'][run_number])
# Separate peak distances by voltage
voltages = df['voltage'].unique()

for voltage in voltages:
    voltage_df = df[df['voltage'] == voltage]

    # Plot peak distances for the current voltage
    plt.figure(figsize=(10, 6))
    plt.scatter(voltage_df.index, voltage_df['peak_distance_1_2'], label='Peak 1-2 Distance')
    plt.scatter(voltage_df.index, voltage_df['peak_distance_2_3'], label='Peak 2-3 Distance')
    plt.xlabel('Index')
    plt.ylabel('Peak Distance')
    plt.legend()
    plt.title(f'Peak Distances at Voltage: {voltage}')
    plt.grid(True)
    plt.show()

