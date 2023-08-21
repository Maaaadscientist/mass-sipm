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
        file2 = ROOT.TFile(f"{filepath_lightmap}/main_run_0162/maps_run162.root")
        hname = "light_map_full"
        hname2 = "reff_mu_1D"
        reff_x = []
        reff_y = []
        reff_z = []
        reff_z_var = []
        for po in range(16):
        
            light_hist = file1.Get(hname + f"/{hname}_tile{po}")
            light_hist_good = file2.Get(hname + f"/{hname}_tile{po}")
            reff_hist = file1.Get(hname2 + f"/reff_mu_1D_tile{po}")
            x_po, y_po = get_coordinates_4x4(po + 1)
            x_ref, y_ref = convert_coordinates_4x4(x_po, y_po, float(ref_offset_x), float(ref_offset_y), yaml_data)
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
                mu_good = light_hist_good.GetBinContent(x_pt, y_pt)
                z_good_list.append(mu_good / pde)
        # Create the interpolation/extrapolation function
        rbf = Rbf(x_list, y_list, z_list, function='linear')
        rbf_good = Rbf(x_list, y_list, z_good_list, function='linear')
        
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
        offset_range = np.linspace(-100, 100, 2001)
        
        # Create a 2D mesh grid of offsets
        dx, dy = np.meshgrid(offset_range, offset_range)
        
        # Calculate the chi-squared statistic for each offset
        chi2_values = np.zeros_like(dx)
        chi2_values_good = np.zeros_like(dx)
        for i in range(len(offset_range)):
            for j in range(len(offset_range)):
                x_unknown = initial_x + dx[i, j]
                y_unknown = initial_y + dy[i, j]
                expected_intensities = rbf(x_unknown, y_unknown)
                expected_intensities_good = rbf_good(x_unknown, y_unknown)
                chi2_values[i, j] = np.sum((observed_intensities - expected_intensities) ** 2 )
                chi2_values_good[i, j] = np.sum((observed_intensities - expected_intensities_good) ** 2 )
        # Calculate the 1-sigma and 2-sigma chi-squared thresholds
        chi2_1sigma = 0.064
        chi2_2sigma = 0.256
        # Find the minimum chi-squared value and its position
        min_chi2 = np.min(chi2_values)
        min_pos = np.unravel_index(np.argmin(chi2_values), chi2_values.shape)

        min_chi2_good = np.min(chi2_values_good)
        min_pos_good = np.unravel_index(np.argmin(chi2_values_good), chi2_values_good.shape)
        
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
        # Plot the 2D chi-squared profile
        plt.figure(figsize=(8, 6))
        plt.imshow(chi2_values_good, origin='lower', extent=[offset_range.min(), offset_range.max(), offset_range.min(), offset_range.max()], cmap='YlGnBu')
        plt.colorbar(label='$\chi^2$')
        
        contour1 = plt.contour(dx, dy, chi2_values_good, levels=[chi2_1sigma], colors=['mediumslateblue'])
        contour2 = plt.contour(dx, dy, chi2_values_good, levels=[chi2_2sigma], colors=['mediumorchid'])
        plt.plot(0, 0, 'go', label='Original Point')
        plt.text(-16, -10, '(0, 0)', fontsize=8, color='black', fontweight='bold')
        plt.plot(dx[min_pos_good], dy[min_pos_good], 'ro', label=f'Minimum $\chi^2$ ({min_chi2_good:.2f})')
        plt.text(dx[min_pos_good] + 3, dy[min_pos_good] + 3, f'({dx[min_pos_good]: 0.2f}, {dy[min_pos_good]: 0.2f})', fontsize=8, color='firebrick', fontweight='bold')
        
        # Set the legend labels for the contours
        plt.clabel(contour1, inline=1, fontsize=10, fmt={chi2_1sigma:r'$\bar{\Delta\mu} = 0.01 $'})
        plt.clabel(contour2, inline=1, fontsize=10, fmt={chi2_2sigma:r'$\bar{\Delta\mu} = 0.02 $'})
        
        plt.xlabel('$\Delta x (mm)$')
        plt.ylabel('$\Delta y (mm)$')
        plt.title(f'2D $\chi^2$ Profile of Main-run {run_number} with light-run 127 ')
        plt.legend()
        plt.savefig(f"{filepath}/{name_without_extension}_chi2profile_centrallycorrected.pdf")
        plt.clf()

        light_map_uncorrected = np.empty((16, 16), dtype=float)         
        light_map_corrected = np.empty((16, 16), dtype=float)         
        light_map_centrallycorrected = np.empty((16, 16), dtype=float)         
        for po in range(16):
        
            x_po, y_po = get_coordinates_4x4(po + 1)
            
            for ch in range(1,17):
                x_ch, y_ch = get_coordinates_4x4(ch)
                light_intensity_uncorrected = 0.
                light_intensity_corrected = 0.
                light_intensity_centrallycorrected = 0.
                for pair in convert_1to4(x_ch,y_ch):
                    x_pt = pair[0]
                    y_pt = pair[1]
                    x_real, y_real = convert_coordinates_8x8(x_pt, y_pt, yaml_data)
                    x_real, y_real = convert_coordinates_4x4(x_po, y_po, x_real, y_real, yaml_data)
                    x_corr = x_real + dx[min_pos]
                    y_corr = y_real + dy[min_pos]
                    x_good = x_real + dx[min_pos_good]
                    y_good = y_real + dy[min_pos_good]
             
                    avr_light_uncorr = average_light_intensity_square(x_real, y_real, 6, rbf)
                    avr_light_corr = average_light_intensity_square(x_corr, y_corr, 6, rbf)
                    avr_light_centrallycorr = average_light_intensity_square(x_good, y_good, 6, rbf_good)
                    light_intensity_uncorrected += avr_light_uncorr
                    light_intensity_corrected += avr_light_corr
                    light_intensity_centrallycorrected += avr_light_centrallycorr
                light_map_uncorrected[po][ch-1] = light_intensity_uncorrected
                light_map_corrected[po][ch-1] = light_intensity_corrected
                light_map_centrallycorrected[po][ch-1] = light_intensity_centrallycorrected

        ov_dict = np.empty((16, 16, 6), dtype=float)
        for po in range(16):
            hist_vbd = file1.Get(f"vbd/vbd_tile{po}")
            for ch in range(1, 17):
                x, y = get_coordinates_4x4(ch)
                #print(po, ch, hist_vbd.GetBinContent(x,y))
                for ov in range(1, 7):
                    ov_dict[po][ch-1][ov-1] = ov - hist_vbd.GetBinContent(x,y) 
        
        mu_dict = np.empty((16, 16, 6), dtype=float)
        dcr_dict = np.empty((16, 16, 6), dtype=float)
        for po in range(16):
            for ov in range(1, 7):
                hist_sipm_mu = file1.Get(f"sipm_mu/sipm_mu_ov{ov}_tile{po}")
                hist_dcr = file1.Get(f"dcr/dcr_ov{ov}_tile{po}")
                for ch in range(1, 17):
                    x, y = get_coordinates_4x4(ch)
                    mu_dict[po][ch - 1][ov-1] = hist_sipm_mu.GetBinContent(x,y)
                    dcr_dict[po][ch - 1][ov-1] = hist_dcr.GetBinContent(x,y)
                
        target_ov = 4.0
        mu_ov4V_dict = np.empty((16, 16), dtype=float)
        pde_map_uncorrected = np.empty((16, 16), dtype=float)         
        pde_map_corrected = np.empty((16, 16), dtype=float)         
        pde_map_centrallycorrected = np.empty((16, 16), dtype=float)         
        for position in range(16):
            for channel in range(16):
                ov_values = ov_dict[position, channel, :]  # Overvoltage values (0, 1, 2, 3, 4, 5)
                mu_values = mu_dict[position, channel, :]  # Mu values for the current position and channel
                dcr_values = dcr_dict[position, channel, :]  # Mu values for the current position and channel
                
                # Perform linear interpolation
                interpolated_mu = np.interp(target_ov, ov_values, mu_values)
                interpolated_dcr = np.interp(target_ov, ov_values, dcr_values)
                
                # Store the interpolated value in mu_ov3V_dict
                mu_ov4V_dict[position, channel] = interpolated_mu
                pde_map_uncorrected[position, channel] = interpolated_mu / light_map_uncorrected[position, channel]
                pde_map_corrected[position, channel] = interpolated_mu / light_map_corrected[position, channel]
                pde_map_centrallycorrected[position, channel] = interpolated_mu / light_map_centrallycorrected[position, channel]
                #print(position, channel+1, interpolated_dcr)



        x = np.arange(16)       
        fig3, ax3 = plt.subplots()
        ax3.axhline(y=0.44, color='red', linestyle='--', label = 'Min 44%')
        ax3.axhline(y=0.47, color='blue', linestyle='--', label = 'Typical 47%')
        for ch in range(16):
            ax3.scatter(x, pde_map_uncorrected[:, ch], label=f'Channel {ch+1}', s=5)
            
              
        # Set labels and title
        ax3.set_xlabel('SiPM Tile number')
        ax3.set_ylabel('PDE')
        ax3.set_title(f'Uncorrected Photon Detection Efficiencies (Run {run_number})')
        
        # Add a horizontal line at y = 0.5
        #ax2.axhline(y=41.7, color='red', linestyle='--', label = 'Max 41.7')
        #ax2.axhline(y=13.9, color='blue', linestyle='--', label = 'Typical 13.9')
        # Add a horizontal line at y = 0.5
        # Set x-axis ticks and labels
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(i) for i in range(16)])
        
        # Set grid lines on the y-axis
        ax3.grid(axis='y')
        # Set the y-axis limits
        ax3.set_ylim(0.2, 0.8)
        # Add a legend
        #ax.legend()
        legend3 = ax3.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
        
        # Adjusting the plot to accommodate the legend
        plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards
        plt.savefig(f"{filepath}/pde_tiles_uncorrected_run{run_number}.pdf")
        plt.clf()
        fig4, ax4 = plt.subplots()
        ax4.axhline(y=0.44, color='red', linestyle='--', label = 'Min 44%')
        ax4.axhline(y=0.47, color='blue', linestyle='--', label = 'Typical 47%')
        for ch in range(16):
            ax4.scatter(x, pde_map_corrected[:, ch], label=f'Channel {ch+1}', s=5)
            
              
        # Set labels and title
        ax4.set_xlabel('SiPM Tile number')
        ax4.set_ylabel('PDE')
        ax4.set_title(f'Position-corrected Photon Detection Efficiencies (Run {run_number})')
        
        # Add a horizontal line at y = 0.5
        #ax2.axhline(y=41.7, color='red', linestyle='--', label = 'Max 41.7')
        #ax2.axhline(y=13.9, color='blue', linestyle='--', label = 'Typical 13.9')
        # Add a horizontal line at y = 0.5
        # Set x-axis ticks and labels
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(i) for i in range(16)])
        
        # Set grid lines on the y-axis
        ax4.grid(axis='y')
        # Set the y-axis limits
        ax4.set_ylim(0.2, 0.8)
        # Add a legend
        #ax.legend()
        legend4 = ax4.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
        
        # Adjusting the plot to accommodate the legend
        plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards
        plt.savefig(f"{filepath}/pde_tiles_corrected_run{run_number}.pdf")
        plt.clf()

        fig5, ax5 = plt.subplots()
        ax5.axhline(y=0.44, color='red', linestyle='--', label = 'Min 44%')
        ax5.axhline(y=0.47, color='blue', linestyle='--', label = 'Typical 47%')
        for ch in range(16):
            ax5.scatter(x, pde_map_centrallycorrected[:, ch], label=f'Channel {ch+1}', s=5)
            
              
        # Set labels and title
        ax5.set_xlabel('SiPM Tile number')
        ax5.set_ylabel('PDE')
        ax5.set_title(f'Position-corrected Photon Detection Efficiencies (Run {run_number})')
        
        # Add a horizontal line at y = 0.5
        #ax2.axhline(y=41.7, color='red', linestyle='--', label = 'Max 41.7')
        #ax2.axhline(y=13.9, color='blue', linestyle='--', label = 'Typical 13.9')
        # Add a horizontal line at y = 0.5
        # Set x-axis ticks and labels
        ax5.set_xticks(x)
        ax5.set_xticklabels([str(i) for i in range(16)])
        
        # Set grid lines on the y-axis
        ax5.grid(axis='y')
        # Set the y-axis limits
        ax5.set_ylim(0.2, 0.8)
        # Add a legend
        #ax.legend()
        legend5 = ax5.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))
        
        # Adjusting the plot to accommodate the legend
        plt.subplots_adjust(right=0.7)  # Increase the value to move the legend leftwards
        plt.savefig(f"{filepath}/pde_tiles_centrallycorrected_run{run_number}.pdf")
        plt.clf()
