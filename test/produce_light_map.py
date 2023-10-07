import csv
import yaml
import numpy as np
from scipy.interpolate import Rbf
#import matplotlib.pyplot as plt  # If you want to visualize
import ROOT
from pykrige.ok import OrdinaryKriging

from collections import defaultdict

def get_kriging_prediction(x_point, y_point, x_data, y_data, z_data, variogram_model='linear'):
    z_intensities = z_data[:, 0]
    OK = OrdinaryKriging(x_data, y_data, z_intensities, variogram_model=variogram_model, verbose=False, enable_plotting=False)
    z_pred, z_var = OK.execute('points', x_point, y_point)
    z_std = np.sqrt(z_var)
    return z_pred[0], z_std[0]

def convert_coordinates_8x8(x_index, y_index, yaml_data):
    original_x = yaml_data['original_x']
    original_y = yaml_data['original_y']
    offset_x_nogap = yaml_data['points_offset_x_nogap']
    offset_x_gap1 = yaml_data['points_offset_x_gap1']
    offset_x_gap2 = yaml_data['points_offset_x_gap2']
    offset_y_gap = yaml_data['points_offset_y_gap']
    y = - (y_index - 1) * offset_y_gap
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

def get_points_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
    
    points = []
    for x_index in range(1, 9):
        for y_index in range(1, 9):
            x, y = convert_coordinates_8x8(x_index, y_index, yaml_data)
            points.append(((x,y), (x_index, y_index)))
    return points

def ch_to_8x8(ch):
    y = ch // 4 + 1 if ch % 4 != 0 else ch // 4 
    x = ch % 4 if ch % 4 != 0 else 4 
    return [(2*x-1, 2*y-1), (2*x, 2*y-1), (2*x-1, 2*y), (2*x, 2*y)]

def combine_uncertainties_in_quadrature(uncertainties):
    return np.sqrt(sum([u**2 for u in uncertainties]))

def predict_for_predefined_points(run_type, run_id):
    structured_data = loading_structured_data("timeline.csv", "mu_timeline.csv")
    base_coords_indices = get_points_from_yaml("../config/design-parameters2.yaml")
    query_coords_indices = get_points_from_yaml("../config/design-parameters.yaml")
    
    csv_lines = "run,pos,ch,ref_mu,ref_mu_err\n"
    for pos in range(1,17):
        points = structured_data[run_type][run_id][pos]
        x_data = np.array([coord[0] for coord in points.keys()])
        y_data = np.array([coord[1] for coord in points.keys()])
        z_data = np.array(list(points.values()))

        base_uncertainties = {}
        mu_unc_dict = {}
        for (x_query, y_query), (x_index, y_index) in base_coords_indices:
            _, z_std = get_kriging_prediction(round(x_query,1), round(y_query,1), x_data, y_data, z_data)
            base_uncertainties[(x_index, y_index)] = z_std
            mu_unc = points[(x_query, y_query)]
            mu_unc_dict[(x_index, y_index)] = mu_unc

        query_uncertainties = {}
        for (x_query, y_query), (x_index, y_index) in query_coords_indices:
            z_pred, z_std = get_kriging_prediction(round(x_query,1), round(y_query,1), x_data, y_data, z_data)
            corrected_std = z_std - base_uncertainties[(x_index, y_index)]
            
            query_uncertainties[(x_index, y_index)] = (z_pred, corrected_std)
            
        for ch in range(1, 17):  # channels from 1 to 16
            points_for_ch = ch_to_8x8(ch)
            z_values = []
            uncertainties = []
            for point in points_for_ch:
                x_index, y_index = point
                z_pred, corrected_std = query_uncertainties[(x_index, y_index)]
                mu_std = mu_unc_dict[(x_index, y_index)]
                z_values.append(z_pred)
                uncertainties.append(np.sqrt(corrected_std**2 + mu_std[1]**2))
                #uncertainties.append(corrected_std)
            
            avg_z = sum(z_values)
            combined_uncertainty = combine_uncertainties_in_quadrature(uncertainties) # A simple average for now, can be adjusted if needed
            csv_lines += f"{run_id},{pos},{ch},{avg_z},{combined_uncertainty}\n"
            print(f"Channel {ch}: Predicted z = {avg_z}, Combined Uncertainty = {combined_uncertainty}")
        with open(f"reffmu_{run_type}_run_{run_id}.csv", "w") as csv_file:
            csv_file.write(csv_lines)

def loading_structured_data(timeline_path, mu_path):
    timeline_data = read_csv(timeline_path)
    mu_timeline_data = read_csv(mu_path)

    #structured_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    structured_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))


    for mu_data in mu_timeline_data:
        nPoints = int(mu_data['nPoints'])
        if nPoints < 8:
            continue


        x, y = float(mu_data['x']), float(mu_data['y'])
        pos = int(mu_data['pos'])
        slope = float(mu_data['slope'])
        intercept = float(mu_data['intercept'])
        mu_unc = float(mu_data['mu_unc'])  # Extracting the uncertainty
        for t_data in timeline_data:
            run_id = t_data['id']
            run_type = t_data['type']
            total_time = float(t_data['total_time'])

            light_intensity = compute_light_intensity(total_time, slope, intercept)

            structured_data[run_type][run_id][pos][(x, y)] = (light_intensity, mu_unc)
    # Sorting the structured_data by pos in ascending order
    for run_type, runs in structured_data.items():
        for run_id, pos_dict in runs.items():
            sorted_pos_dict = dict(sorted(pos_dict.items()))
            structured_data[run_type][run_id] = sorted_pos_dict


    return structured_data

def krige_interpolation(x, y, z, x_grid, y_grid, variogram_model='linear'):
    """
    Perform Kriging interpolation.
    
    Parameters:
    - x, y, z : Arrays of data points.
    - x_grid, y_grid : Grid on which to interpolate.
    - variogram_model : String specifying which variogram model to use.
    
    Returns:
    - z_kriged : 2D array of interpolated values on the grid.
    - z_std_dev: 2D array of standard deviations of interpolated values on the grid.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    z = np.array(z, dtype=np.float64)
    x_grid = np.array(x_grid, dtype=np.float64)
    y_grid = np.array(y_grid, dtype=np.float64)
    OK = OrdinaryKriging(x, y, z, variogram_model=variogram_model, verbose=False, enable_plotting=False)
    z_kriged, z_var = OK.execute('grid', x_grid, y_grid)
    z_std_dev = np.sqrt(z_var)  # standard deviation is the square root of variance
    return z_kriged, z_std_dev

def krige_interpolation_single(x, y, z, x_new, y_new, variogram_model='linear'):
    """
    Perform Kriging interpolation.
    
    Parameters:
    - x, y, z : Arrays of data points.
    - x_grid, y_grid : Grid on which to interpolate.
    - variogram_model : String specifying which variogram model to use.
    
    Returns:
    - z_kriged : 2D array of interpolated values on the grid.
    - z_std_dev: 2D array of standard deviations of interpolated values on the grid.
    """
    x = np.array(x, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    z = np.array(z, dtype=np.float64)
    OK = OrdinaryKriging(x, y, z, variogram_model=variogram_model, verbose=False, enable_plotting=False)
    z_kriged, z_var = OK.execute('grid', x_new, y_new)
    z_std_dev = np.sqrt(z_var)  # standard deviation is the square root of variance
    return z_kriged, z_std_dev

def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def compute_light_intensity(total_time, slope, intercept):
    return intercept + slope * total_time

def interpolate_and_store(run_type, run_id, method='krige', variogram_model='linear'):

    structured_data = loading_structured_data("timeline.csv", "mu_timeline.csv")
    #return structured_data
    # Generate the finer grid
    #x_fine = np.linspace(-5, 55, 60)  # 500 is an example. You can choose any resolution you want
    #y_fine = np.linspace(5, -45, 50)
    #X, Y = np.meshgrid(x_fine, y_fine)
    # Ranges for extrapolation
    x_start, x_end, x_step = -5, 65, 0.5
    y_start, y_end, y_step = -65, 5, 0.5

    #interpolated_data = {}

    #for run_type, runs in structured_data.items():
    #    interpolated_data[run_type] = {}
    #    for run_id, positions in runs.items():
    #        interpolated_data[run_type][run_id] = {}
    # Initialize the ROOT file for storing TGraph2D objects

    filename = f"preciseMap_{run_type}_run_{run_id}.root"
    file_lightmap = ROOT.TFile(filename, "recreate")
    x_grid = np.arange(x_start, x_end + x_step, x_step)
    y_grid = np.arange(y_start, y_end + y_step, y_step)
    #x_grid = np.array([9.07, 15.07])
    #y_grid = np.array([0, -6])
    positions = structured_data[run_type][run_id]
    for pos in range(1,17):
        points = positions[pos]

        x = np.array([coord[0] for coord in points.keys()])
        y = np.array([coord[1] for coord in points.keys()])
        #z = np.array(list(points.values()))
        z_values, z_uncertainties_original = zip(*points.values())  # Unpacking the 2D structure
        z = np.array(z_values)
        # Choose interpolation method
        if method == 'krige':

            # Using Kriging for interpolation and getting values and uncertainties
            z_interp, z_uncertainties = krige_interpolation(x, y, z, x_grid, y_grid, variogram_model)

        # elif method == 'some_other_method':
        #     z_interp = some_other_interpolation_function(...)
        else:
            raise ValueError(f"Interpolation method '{method}' not recognized!")

        if pos == 8:
            for k in range(len(x)):
                single_z, single_z_err = krige_interpolation_single(x, y, z, x[k], y[k])
        graph_name = f"graph_pos_{pos}"
        graph2 = ROOT.TGraph2D()  # Initialize with total number of points
        graph2.SetNameTitle(graph_name, graph_name)

        point_index = 0
        for i, xi in enumerate(x_grid):
            for j, yi in enumerate(y_grid):
                #if i < z_interp.shape[0] and j < z_interp.shape[1]:  # Ensure we are within bounds
                #single_z, single_z_err = krige_interpolation_single(x, y, z, xi, yi)
                zi = z_interp[j, i]
                graph2.SetPoint(point_index, xi, yi, zi)
                point_index += 1

        # Write the TGraph2D object to the ROOT file
        graph2.Write(graph_name)
        del graph2  # Ensure proper cleanup
        # Save uncertainties
        graph_name_unc = f"graph_uncertainties_pos_{pos}"
        graph_uncertainties = ROOT.TGraph2D()  
        graph_uncertainties.SetNameTitle(graph_name_unc, graph_name_unc)
        point_index = 0
        for i, xi in enumerate(x_grid):
            for j, yi in enumerate(y_grid):
                if i < z_interp.shape[0] and j < z_interp.shape[1]:  # Ensure we are within bounds
                    zi_unc = z_uncertainties[j, i]
                    graph_uncertainties.SetPoint(point_index, xi, yi, zi_unc)
                    point_index += 1
        graph_uncertainties.Write(graph_name_unc)

    file_lightmap.Close()

if __name__ == "__main__":
    #run_type_input = input("Enter run type (light/main): ")
    #run_id_input = input("Enter run id: ")
    #interpolate_and_store(run_type_input, run_id_input, 'krige', 'linear')
    run_type_input = input("Enter run type (light/main): ")
    run_id_input = input("Enter run id: ")
    predict_for_predefined_points(run_type_input, run_id_input)
    interpolate_and_store(run_type_input, run_id_input, 'krige', 'linear')





