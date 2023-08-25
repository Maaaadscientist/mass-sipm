import yaml
import os, sys
import re
import ROOT
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import Rbf
from scipy.stats import chi2
from scipy.stats import linregress
from scipy.stats import zscore



def get_coordinates_8x8(ref_number):
    ref_number -= 1
    if 0 <= ref_number < 64:  # Valid ref_numbers are between 0 and 63
        # Calculate row (y)
        y_quotient, y_remainder = divmod(ref_number, 8)
        y = 8 - y_quotient

        # Calculate column (x)
        if y % 2 == 1:  # Odd row
            x = (8 - y_remainder) % 8
            if x == 0:
                x = 8
        else:  # Even row
            x = y_remainder + 1

        return x, y
    else:
        return None  # Reference number is out of bounds

def get_coordinates_4x4(ref_number):
    ref_number -= 1
    if 0 <= ref_number < 16:  # Valid ref_numbers are between 0 and 63
        # Calculate row (y)
        y_quotient, y_remainder = divmod(ref_number, 4)
        y = 4 - y_quotient
        x = y_remainder + 1

        return x, y
    else:
        return None  # Reference number is out of bounds

def convert_coordinates_8x8(x_index, y_index, yaml_data):
    original_x = yaml_data['original_x']
    original_y = yaml_data['original_y']
    offset_x_nogap = yaml_data['points_offset_x_nogap']
    offset_x_gap1 = yaml_data['points_offset_x_gap1']
    offset_x_gap2 = yaml_data['points_offset_x_gap2']
    offset_y_gap = yaml_data['points_offset_y_gap']
    y = (y_index - 1) * offset_y_gap
    if x_index == 1:
        x = 0
    elif x_index == 2:
        x = offset_x_nogap
    elif x_index == 3:
        x = offset_x_nogap * 1 + offset_x_gap1 * 1
    elif x_index == 4:
        x = offset_x_nogap * 2 + offset_x_gap1 * 1
    elif x_index == 5:
        x = offset_x_nogap * 2 + offset_x_gap1 * 1 + offset_x_gap2 * 1
    elif x_index == 6:
        x = offset_x_nogap * 3 + offset_x_gap1 * 1 + offset_x_gap2 * 1
    elif x_index == 7:
        x = offset_x_nogap * 3 + offset_x_gap1 * 2 + offset_x_gap2 * 1
    elif x_index == 8:
        x = offset_x_nogap * 4 + offset_x_gap1 * 2 + offset_x_gap2 * 1

    x += original_x
    y += original_y

    return x, y

def convert_coordinates_4x4(x_po, y_po, x, y, yaml_data, original = (0 , 0)):
    tile_offset_y = yaml_data['tile_offset_y']
    tile_offset_x = yaml_data['tile_offset_x']
    x_real = x + (x_po - 1) * tile_offset_x - original[0]
    y_real = y + (y_po - 1) * tile_offset_y - original[1]
    return x_real, y_real

def convert_1to4(x,y):
    return [(2*x-1, 2*y-1), (2*x, 2*y-1), (2*x-1, 2*y), (2*x, 2*y)]

def average_light_intensity_square(x_center, y_center, square_length, rbf):
    # Calculate the coordinates for the square's corners
    x_min = x_center - square_length / 2
    x_max = x_center + square_length / 2
    y_min = y_center - square_length / 2
    y_max = y_center + square_length / 2

    # Specify the coordinates where you want to sample light intensity
    x_sample = np.linspace(x_min, x_max, 100)  # Adjust the number of samples as needed
    y_sample = np.linspace(y_min, y_max, 100)  # Adjust the number of samples as needed

    # Create a meshgrid for sampling
    X_sample, Y_sample = np.meshgrid(x_sample, y_sample)

    # Sample the light intensity data within the square
    z_sample = rbf(X_sample, Y_sample)

    # Calculate the average light intensity within the square
    average_intensity = np.mean(z_sample)

    return average_intensity
def find_root_files(start_path):
    """
    Recursively find all .root files from the start_path.

    :param start_path: Directory or file path to start the search from.
    :return: A list of .root file paths.
    """
    root_files = []

    # Check if the provided path is a directory or a file
    if os.path.isdir(start_path):
        for root, _, files in os.walk(start_path):
            for file in files:
                if file.endswith('.root'):
                    full_path = os.path.join(root, file)
                    root_files.append(full_path)
    elif os.path.isfile(start_path) and start_path.endswith('.root'):
        root_files.append(start_path)
    root_files.sort()

    return root_files
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_directory_or_file>")
        sys.exit(1)

    path = sys.argv[1]
    root_files = find_root_files(path)
    # Specify the path to your YAML file
    yaml_file_path = 'design-parameters.yaml'
    reff_config_path = 'reff-pde.yaml'
    
    # Read and parse the YAML file
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    with open(reff_config_path, 'r') as yaml_file:
        reff_data = yaml.safe_load(yaml_file)
    # Now you can access the data in the YAML file as a dictionary
    ref_offset_x = yaml_data['ref_offset_x']
    ref_offset_y = yaml_data['ref_offset_y']
    length_tile = yaml_data['length_tile']
    length_ch = yaml_data['length_ch']
    length_half_ch = yaml_data['length_half_ch']
    logfile = open("completed.txt")
    completed_run = [line.rstrip('\n') for line in logfile.readlines()]
    
    num_tiles = 16
    num_time_points = 115
    
    data = np.zeros((num_tiles, num_time_points))
    i = 0
    for file in root_files:
        filepath = "/".join(os.path.abspath(file).split("/")[0:-1])
        filepath_lightmap = "/".join(os.path.abspath(file).split("/")[0:-2])
        filename = os.path.abspath(file).split("/")[-1]  # Extracts "xxx.root"
        name_without_extension = filename.split(".")[0]  # Extracts "xxx"
        run_number = re.findall(r'\d+', name_without_extension)[0]
        # ... and so on for other variables
        
        file1 = ROOT.TFile(f"{file}")
        file2 = ROOT.TFile(f"{filepath_lightmap}/main_run_0162/maps_run162.root")
        hname = "light_map_full"
        hname2 = "reff_mu_1D"
        for po in range(num_tiles):
        
            reff_hist = file1.Get(hname2 + f"/reff_mu_1D_tile{po}")
            if not reff_hist:
                continue
            pde = reff_data.get(po)
            reff_light_intensity = reff_hist.GetMean()/pde
            reff_stddev = reff_hist.GetRMS()/pde
            data[po][i] = reff_light_intensity
            print(i, reff_light_intensity)
        i+= 1
    # Replace this with your actual 2D array
    # For example, let's create a random 2D array with 5 tiles and 10 time points
    
    # Plot the scatter points and outlier-removed linear fit for each tile separately
    for tile_idx in range(num_tiles):
        plt.figure(figsize=(8, 4))  # Adjust the figure size as needed
    
        x = np.arange(num_time_points)  # Time points
        y = data[tile_idx]              # Values for the current tile
    
        # Calculate Z-scores for the y values
        z_scores = zscore(y)
        threshold = 2.0  # Adjust this threshold to control outlier removal
        
        # Filter out outliers
        filtered_indices = np.abs(z_scores) < threshold
        x_filtered = x[filtered_indices]
        y_filtered = y[filtered_indices]
        
        plt.scatter(x_filtered, y_filtered, label=f'Tile {tile_idx}', color='blue')
        
        # Perform linear fit on the filtered data
        slope, intercept, r_value, p_value, std_err = linregress(x_filtered, y_filtered)
        fit_line = slope * x + intercept
        plt.plot(x, fit_line, color='red', label='Linear Fit')
        
        plt.title(f'Tile {tile_idx} - Linear Fit (Outliers Removed)')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f"tile{tile_idx}_stability.pdf")
