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


if __name__ == '__main__':
    # Read and parse the YAML file
    yaml_file_path = 'config/design-parameters2.yaml'
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    # Now you can access the data in the YAML file as a dictionary
    ref_offset_x = yaml_data['ref_offset_x']
    ref_offset_y = yaml_data['ref_offset_y']
    length_tile = yaml_data['length_tile']
    length_ch = yaml_data['length_ch']
    length_half_ch = yaml_data['length_half_ch']
    
    # ... and so on for other variables
    
    x_list = []
    y_list = []
    po_list = []
    pt_list = []
    ref_x_list = []
    ref_y_list = []
    ref_po_list = []
    for po in range(16):
        x_po, y_po = get_coordinates_4x4(po + 1)
        x_ref, y_ref = convert_coordinates_4x4(x_po, y_po, float(ref_offset_x), float(ref_offset_y), yaml_data)
        x_list.append(x_ref)
        y_list.append(y_ref)
        po_list.append(po)
        pt_list.append(0)

        ref_x_list.append(x_ref + 10)
        ref_y_list.append(y_ref + 5)
        ref_po_list.append(po)
        
        for point in range(1,65):
            x_pt, y_pt = get_coordinates_8x8(point)
            x_relative, y_relative = convert_coordinates_8x8(x_pt, y_pt, yaml_data)
            x_real, y_real = convert_coordinates_4x4(x_po, y_po, x_relative, y_relative, yaml_data)
            x_list.append(x_real)
            y_list.append(y_real)
            po_list.append(po)
            pt_list.append(point)
    square_length = 6  # Adjust the length of the square's side as needed
    
    plt.figure(figsize=(12, 8))  # Enlarged figure size
    
    #plt.scatter(x_list, y_list, color='blue', marker='o', s=100)
    
    for i, (po, pt) in enumerate(zip(po_list, pt_list)):
        plt.text(x_list[i], y_list[i] - 2, f'{pt}', fontsize=6, ha='center', va='bottom')
    
        # Calculate the coordinates for the square's lower-left corner
        square_x = x_list[i] - square_length / 2
        square_y = y_list[i] - square_length / 2
    
        # Create a Rectangle patch and add it to the plot
        square = Rectangle((square_x, square_y), square_length, square_length, edgecolor='red', fill=False)
        plt.gca().add_patch(square)
    for i in range(16):
        plt.text(ref_x_list[i], ref_y_list[i], f'tile{i}', fontsize=10, ha='center', va='bottom')
    
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Reference SiPM positions')
    
    plt.axis('equal')
    
    plt.savefig(f"positions.pdf")
    plt.clf()
