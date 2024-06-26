import yaml
from array import array
import os, sys
import re
import ROOT
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import Rbf
from scipy.stats import chi2
import pandas as pd


def get_new_coordinates(csv_path, point_index):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Validate point index
    if point_index < 1 or point_index > 64:
        raise ValueError("Point index must be between 1 and 64")
    
    # Filter the DataFrame to get the row corresponding to the point index
    row = df[df['point'] == point_index].iloc[0]
    
    # Extract new_x and new_y values
    new_x = row['new_x']
    new_y = row['new_y']
    
    return (new_x, new_y)


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
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <path_to_directory_or_file> <csv_file>")
        sys.exit(1)

    path = sys.argv[1]
    csv_path = sys.argv[2]
    root_files = find_root_files(path)
    # Specify the path to your YAML file
    yaml_file_path = 'config/design-parameters2.yaml'
    reff_config_path = 'config/reff-pde.yaml'
    
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
    
    
    deviation_x = []
    deviation_y = []
    deviation_linked_x = []
    deviation_linked_y = []
    best_pos_light_intensity = []
    valid_run_list = []

    invalid_run_list = []

        
    for file in root_files:
        filepath = "/".join(os.path.abspath(file).split("/")[0:-1])
        filepath_lightmap = "/".join(os.path.abspath(file).split("/")[0:-2])
        print(filepath)
        filename = os.path.abspath(file).split("/")[-1]  # Extracts "xxx.root"
        name_without_extension = filename.split(".")[0]  # Extracts "xxx"
        run_number = re.findall(r'\d+', name_without_extension)[0]
        # ... and so on for other variables
        
        x_list = []
        y_list = []
        po_list = []
        pt_list = []
        z_list = []
        z_good_list = []
        file1 = ROOT.TFile(f"{file}")
        hname = "light_map_full"
        reff_num = []
        reff_x = []
        reff_y = []
        reff_z = []
        reff_z_var = []
        err_flag = False
        for po in range(16):
        
            light_hist = file1.Get(hname + f"/{hname}_tile{po}")
            x_po, y_po = get_coordinates_4x4(po + 1)
            x_ref, y_ref = convert_coordinates_4x4(x_po, y_po, float(ref_offset_x), float(ref_offset_y), yaml_data)
            pde = reff_data.get(po)
            reff_num.append(po)
            reff_x.append(x_ref)
            reff_y.append(y_ref)
            
            saved_points = []
            for point in range(1,65):
                x_pt, y_pt = get_coordinates_8x8(point)
                x_relative, y_relative = get_new_coordinates(csv_path, point)
                double_count = False
                for saved_point in saved_points:
                    if abs(x_relative - saved_point[0]) < 0.1 and abs(y_relative - saved_point[1]) < 0.1:
                        double_count = True
                if double_count:
                    continue
                if po == 1:
                    print(x_relative, y_relative)
                x_relative = round(x_relative, 1)
                y_relative = round(y_relative, 1)

                #print(x_relative1, x_relative, y_relative1, y_relative)
                x_real, y_real = convert_coordinates_4x4(x_po, y_po, x_relative, y_relative, yaml_data)
                x_list.append(x_real)
                y_list.append(y_real)
                po_list.append(po)
                pt_list.append(point)
                mu = light_hist.GetBinContent(x_pt, y_pt)
                z_list.append(mu / pde)
                saved_points.append((x_relative, y_relative))
        for i in range(len(reff_x)):
            print(reff_num[i], round(reff_x[i],2), round(reff_y[i],2))
        # Create the TGraph2D light map
        x_array = array('d', x_list)
        y_array = array('d', y_list)
        z_array = array('d', z_list)
        n_points = len(x_list)
        # Create the interpolation/extrapolation function
        rbf = Rbf(x_list, y_list, z_list, function='cubic')
        

        # Coordinate ranges and steps
        x_start, x_end, x_step = -100, 400, 0.5  # You can adjust the step to whatever you want
        y_start, y_end, y_step = -200, 300, 0.5  # You can adjust the step to whatever you want
        
        point_index = 0  # Counter for adding points to TGraph2D
        
        # Initialize TGraph2D object
        graph2 = ROOT.TGraph2D()
        # Loop through all points one by one
        for x in np.arange(x_start, x_end + x_step, x_step):
            for y in np.arange(y_start, y_end + y_step, y_step):
                x = round(x, 1)
                y = round(y, 1) 
                # Interpolate z value at this single (x, y) point
                z = rbf(x, y)
                
                # Populate the TGraph2D object
                graph2.SetPoint(point_index, x, y, z)
                point_index += 1
        
        file_lightmap = ROOT.TFile("preciseMap.root", "recreate")
        graph2.Write()
        file_lightmap.Close()
        # Perform the interpolation/extrapolation
