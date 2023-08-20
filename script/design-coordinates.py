import yaml
import os, sys
import re
import ROOT
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from scipy.interpolate import Rbf
from scipy.stats import chi2



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
    
    for file in root_files:
        filepath = "/".join(os.path.abspath(file).split("/")[0:-1])
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
        file1 = ROOT.TFile(f"{file}")
        hname = "light_map_full"
        hname2 = "reff_mu_1D"
        reff_x = []
        reff_y = []
        reff_z = []
        reff_z_var = []
        for po in range(16):
        
            light_hist = file1.Get(hname + f"/{hname}_tile{po}")
            reff_hist = file1.Get(hname2 + f"/reff_mu_1D_tile{po}")
            x_po, y_po = get_coordinates_4x4(po + 1)
            x_ref, y_ref = convert_coordinates_4x4(x_po, y_po, float(ref_offset_x), float(ref_offset_y), yaml_data)
            #x_list.append(x_ref)
            #y_list.append(y_ref)
            #po_list.append(po)
            #pt_list.append(0)
            pde = reff_data.get(po)
            reff_light_intensity = reff_hist.GetMean()/pde
            reff_stddev = reff_hist.GetRMS()/pde
            reff_x.append(x_ref)
            reff_y.append(y_ref)
            reff_z.append(reff_light_intensity)
            reff_z_var.append(reff_stddev * reff_stddev)
            
            for point in range(1,65):
                x_pt, y_pt = get_coordinates_8x8(point)
                x_relative, y_relative = convert_coordinates_8x8(x_pt, y_pt, yaml_data)
                x_real, y_real = convert_coordinates_4x4(x_po, y_po, x_relative, y_relative, yaml_data)
                x_list.append(x_real)
                y_list.append(y_real)
                po_list.append(po)
                pt_list.append(point)
                mu = light_hist.GetBinContent(x_pt, y_pt)
                z_list.append(mu / pde)
        square_length = 6  # Adjust the length of the square's side as needed
        
        plt.figure(figsize=(12, 8))  # Enlarged figure size
        
        #plt.scatter(x_list, y_list, color='blue', marker='o', s=100)
        
        for i, (po, pt) in enumerate(zip(po_list, pt_list)):
            plt.text(x_list[i], y_list[i], f'({po}, {pt})', fontsize=3, ha='center', va='bottom')
        
            # Calculate the coordinates for the square's lower-left corner
            square_x = x_list[i] - square_length / 2
            square_y = y_list[i] - square_length / 2
        
            # Create a Rectangle patch and add it to the plot
            square = Rectangle((square_x, square_y), square_length, square_length, edgecolor='red', fill=False)
            plt.gca().add_patch(square)
        
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title('Reference SiPM positions')
        
        plt.axis('equal')
        
        plt.savefig(f"{filepath}/{name_without_extension}_positions.pdf")
        plt.clf()
        # Create the interpolation/extrapolation function
        rbf = Rbf(x_list, y_list, z_list, function='linear')
        
        # Specify the coordinates where you want to interpolate/extrapolate
        x_new = np.linspace(-20, 300, 1280)  # Note that we are extending the range for extrapolation
        y_new = np.linspace(-20, 300, 1280)  # Note that we are extending the range for extrapolation
        
        # Create a meshgrid for plotting
        X_new, Y_new = np.meshgrid(x_new, y_new)
        
        # Perform the interpolation/extrapolation
        z_new = rbf(X_new, Y_new)
        
        # Plot the original data as scatter plot
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(x_list, y_list, c=z_list, cmap='viridis', s=50)
        plt.colorbar(label='Light Intensity')
        plt.title('Original Light Data $\mu_{ref}/$PDE')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.axis('equal')
        
        # Plot the interpolated/extrapolated data as a heat map
        plt.subplot(1, 2, 2)
        plt.imshow(z_new, origin='lower', extent=[-20, 300, -20, 300], aspect='auto', cmap='viridis')
        plt.colorbar(label='Light Intensity')
        plt.title(f'Interpolated/Extrapolated light map of Run-{run_number}')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.axis('equal')
        plt.savefig(f"{filepath}/{name_without_extension}_lightmap.pdf")
        plt.clf()
        
        observed_intensities = np.array(reff_z)
        observed_intensities_var = np.array(reff_z_var)
        initial_x = np.array(reff_x)
        initial_y = np.array(reff_y)
        # Define the range of offsets to scan in the x and y directions
        offset_range = np.linspace(-100, 100, 1000)
        
        # Create a 2D mesh grid of offsets
        dx, dy = np.meshgrid(offset_range, offset_range)
        
        # Calculate the chi-squared statistic for each offset
        chi2_values = np.zeros_like(dx)
        for i in range(len(offset_range)):
            for j in range(len(offset_range)):
                x_unknown = initial_x + dx[i, j]
                y_unknown = initial_y + dy[i, j]
                expected_intensities = rbf(x_unknown, y_unknown)
                chi2_values[i, j] = np.sum((observed_intensities - expected_intensities) ** 2 )
        # Calculate the 1-sigma and 2-sigma chi-squared thresholds
        chi2_1sigma = 0.064
        chi2_2sigma = 0.256
        # Find the minimum chi-squared value and its position
        min_chi2 = np.min(chi2_values)
        min_pos = np.unravel_index(np.argmin(chi2_values), chi2_values.shape)
        
        # Plot the 2D chi-squared profile
        plt.figure(figsize=(8, 6))
        plt.imshow(chi2_values, origin='lower', extent=[offset_range.min(), offset_range.max(), offset_range.min(), offset_range.max()], cmap='YlGnBu')
        plt.colorbar(label='$\chi^2$')
        
        contour1 = plt.contour(dx, dy, chi2_values, levels=[chi2_1sigma], colors=['mediumslateblue'])
        contour2 = plt.contour(dx, dy, chi2_values, levels=[chi2_2sigma], colors=['mediumorchid'])
        plt.plot(0, 0, 'go', label='Original Point')
        plt.text(-16, -10, '(0, 0)', fontsize=8, color='black', fontweight='bold')
        plt.plot(dx[min_pos], dy[min_pos], 'ro', label=f'Minimum $\chi^2$ ({min_chi2:.2f})')
        plt.text(dx[min_pos] + 3, dy[min_pos] + 3, f'({dx[min_pos]: 0.2f}, {dy[min_pos]: 0.2f})', fontsize=8, color='firebrick', fontweight='bold')
        
        # Set the legend labels for the contours
        plt.clabel(contour1, inline=1, fontsize=10, fmt={chi2_1sigma:r'$\bar{\Delta\mu} = 0.01 $'})
        plt.clabel(contour2, inline=1, fontsize=10, fmt={chi2_2sigma:r'$\bar{\Delta\mu} = 0.02 $'})
        
        plt.xlabel('$\Delta x (mm)$')
        plt.ylabel('$\Delta y (mm)$')
        plt.title(f'2D $\chi^2$ Profile of Main-run {run_number}')
        plt.legend()
        plt.savefig(f"{filepath}/{name_without_extension}_chi2profile.pdf")
        plt.clf()
        
#        for po in range(16):
#        
#            x_po, y_po = get_coordinates_4x4(po + 1)
#            
#            for point in range(1,65):
#                x_pt, y_pt = get_coordinates_8x8(point)
#                x_relative, y_relative = convert_coordinates_8x8(x_pt, y_pt, yaml_data)
#                x_real, y_real = convert_coordinates_4x4(x_po, y_po, x_relative, y_relative, yaml_data, (dx[min_pos], dy[min_pos]))
        
