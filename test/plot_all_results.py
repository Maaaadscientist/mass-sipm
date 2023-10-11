import ROOT
import numpy as np
import pandas as pd
#from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline

excluded_list_file = "config/exclude.txt"
amp_factor_file = "config/gain_factors.csv"
merged_root_path = "final_all.root"


# Certain ov points for the plotting 
# 0    :   3.0
# 1    :   3.5
# 2    :   4.0
# 3    :   4.5
# last :   Vop
ov_points = [3.0,3.5,4.0,4.5]
reff_pdes = np.array([
0.535554960303385,
0.536019011766821,
0.536198283782321,
0.536602586153054,
0.531156769240539,
0.536439670457784,
0.548524108673012,
0.537910427257292,
0.534314856449092,
0.535057498593419,
0.5387169029171,
0.538117606850329,
0.531209765907509,
0.550764174454449,
0.557335550682235,
0.530908875868674,
])
# Load dataframe
ROOT.EnableImplicitMT() 
f = ROOT.TFile.Open(merged_root_path)
df = ROOT.RDataFrame("tree", f)
#df = df.Filter("abs(match_x) < 3 and abs(match_y) < 3")
# Define the OV points for interpolation
#ov_points = np.linspace(df.Min("ov"), df.Max("ov"), num=int((df.Max("ov") - df.Min("ov")) / 0.1))
min_ov = 1.0#df.Reduce("std::min(double a, double b)", "ov")
max_ov = 8.5#df.Reduce("std::max(double a, double b)", "ov")
#ov_points = np.linspace(min_ov, max_ov, num=int((max_ov - min_ov) / 0.1))


def nearest_index(arr, target):
    if target >= arr[np.argmax(arr)]:
        return np.argmax(arr) 
    elif target <= arr[np.argmin(arr)]:
        return np.argmin(arr)
    else:
        return np.abs(arr - target).argmin()

def weighted_error(interpolated_vol, original_vol, original_err):
    # If the interpolated_vol is already in original_vol, return the corresponding error
    if interpolated_vol in original_vol:
        return original_err[original_vol.index(interpolated_vol)]

    # Find indices of the nearest two points
    distances = np.abs(np.array(original_vol) - interpolated_vol)
    sorted_indices = np.argsort(distances)
    nearest_idx1 = sorted_indices[0]
    nearest_idx2 = sorted_indices[1]

    # Calculate inverse distance weights
    weight1 = 1 / distances[nearest_idx1]
    weight2 = 1 / distances[nearest_idx2]

    # Weighted average error
    weighted_err = (weight1 * original_err[nearest_idx1] + weight2 * original_err[nearest_idx2]) / (weight1 + weight2)
    
    return weighted_err

# Get unique combinations
unique_runs = np.unique(df.AsNumpy(["run"])["run"])
unique_poss = np.unique(df.AsNumpy(["pos"])["pos"])
unique_chs = np.unique(df.AsNumpy(["ch"])["ch"])
unique_tiles = np.unique(df.AsNumpy(["tsn"])["tsn"])
with open(excluded_list_file, "r") as file1:
    lines = file1.readlines()
    bad_dict = {}
    for aline in lines:
        bad_dict[aline.strip().split()[0]] = aline.strip().split()[1]
# Process data piece-by-piece and append to output file
#output_file = ROOT.TFile("output.root", "RECREATE")
#tree = ROOT.TTree("interpolated_tree", "Interpolated Data")
amp_gain_df = pd.read_csv(amp_factor_file)

for tsn in unique_tiles:
    match_tsn = tsn
    if tsn == 13354:
        tsn = 3354
    if tsn == 31437: 
        tsn = 1437
    write_header = True
    filter_tile_str = f"tsn == {tsn}"
    tile_df = df.Filter(filter_tile_str)
    runs = np.unique(tile_df.AsNumpy(["run"])["run"])
    
    for run in runs:
        tsn_err_flag = False
        if tsn == -1:
            tsn_err_flag = True
            if run == 186:
                tsn = 3241
            elif run == 187:
                tsn = 2694 #?
            elif run == 188:
                tsn = 2981
            elif run == 189:
                tsn = 2997
            elif run == 190:
                tsn = 3017
            elif run == 191:
                tsn = 3038 #?
            elif run == 193:
                tsn = 3057
            elif run == 194:
                tsn = 3085
            elif run == 195:
                tsn = 3101
            elif run == 196:
                tsn = 1839
            elif run == 197:
                tsn = 1856 #?
            elif run == 198:
                tsn = 1964
            elif run == 199:
                tsn = 1774
            elif run == 200:
                tsn = 1719 #?
            elif run == 201:
                tsn = 1747 #?
            elif run == 202:
                tsn = 1887 #?
            elif run == 203:
                tsn = 1907
            elif run == 204:
                tsn = 1925
            elif run == 205:
                tsn = 1943 #?
            elif run == 206:
                tsn = 2682
            elif run == 317:
                tsn = 3801
        if str(match_tsn) in bad_dict.keys():
            if run == int(bad_dict[str(match_tsn)]):
                continue
        run_df = tile_df.Filter(f"run == {run}") 
        unique_vbd = run_df.AsNumpy(["vbd"])["vbd"]
        pos_val = np.unique(run_df.AsNumpy(["pos"])["pos"])[0]
        unique_vbd_err = run_df.AsNumpy(["vbd_err"])["vbd_err"]
        # 1. Find the index of the maximum and minimum values of vbd
        index_max_vbd = np.argmax(unique_vbd)
        index_min_vbd = np.argmin(unique_vbd)
        
        # 2. Access the values and their errors using the indices
        max_vbd = unique_vbd[index_max_vbd]
        min_vbd = unique_vbd[index_min_vbd]
        mean_vbd = np.mean(unique_vbd)
        mean_vbd_err = np.sqrt(np.sum(unique_vbd_err**2) / len(unique_vbd_err)) 
        
        max_vbd_err = unique_vbd_err[index_max_vbd]
        min_vbd_err = unique_vbd_err[index_min_vbd]
        
        # 3. Derive the difference and error
        difference = max_vbd - min_vbd
        
        # For error propagation in subtraction, we use the following formula:
        # Δz = sqrt(Δx^2 + Δy^2), where z = x - y
        error = np.sqrt(max_vbd_err**2 + min_vbd_err**2)
        
        tile_vop = np.empty(16, dtype=float)
        tile_vop_max = 0.
        tile_vop_min = 0.
        tile_vop_mean = 0.
        tile_pde = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_pde_err = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_pde_max = np.empty(len(ov_points)+1, dtype=float)
        tile_pde_max_err = np.empty(len(ov_points)+1, dtype=float)
        tile_pde_min = np.empty(len(ov_points)+1, dtype=float)
        tile_pde_min_err = np.empty(len(ov_points)+1, dtype=float)
        tile_pde_mean = np.empty(len(ov_points)+1, dtype=float)
        tile_pde_mean_err = np.empty(len(ov_points)+1, dtype=float)
        tile_dcr = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_dcr_err = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_dcr_max = np.empty(len(ov_points)+1, dtype=float)
        tile_dcr_max_err = np.empty(len(ov_points)+1, dtype=float)
        tile_dcr_min = np.empty(len(ov_points)+1, dtype=float)
        tile_dcr_min_err = np.empty(len(ov_points)+1, dtype=float)
        tile_dcr_mean = np.empty(len(ov_points)+1, dtype=float)
        tile_dcr_mean_err = np.empty(len(ov_points)+1, dtype=float)
        tile_pct = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_pct_err = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_pct_max = np.empty(len(ov_points)+1, dtype=float)
        tile_pct_max_err = np.empty(len(ov_points)+1, dtype=float)
        tile_pct_min = np.empty(len(ov_points)+1, dtype=float)
        tile_pct_min_err = np.empty(len(ov_points)+1, dtype=float)
        tile_pct_mean = np.empty(len(ov_points)+1, dtype=float)
        tile_pct_mean_err = np.empty(len(ov_points)+1, dtype=float)
        tile_gain = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_gain_err = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_gain_max = np.empty(len(ov_points)+1, dtype=float)
        tile_gain_max_err = np.empty(len(ov_points)+1, dtype=float)
        tile_gain_min = np.empty(len(ov_points)+1, dtype=float)
        tile_gain_min_err = np.empty(len(ov_points)+1, dtype=float)
        tile_gain_mean = np.empty(len(ov_points)+1, dtype=float)
        tile_gain_mean_err = np.empty(len(ov_points)+1, dtype=float)
        tile_eps = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_eps_err = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_eps_max = np.empty(len(ov_points)+1, dtype=float)
        tile_eps_max_err = np.empty(len(ov_points)+1, dtype=float)
        tile_eps_min = np.empty(len(ov_points)+1, dtype=float)
        tile_eps_min_err = np.empty(len(ov_points)+1, dtype=float)
        tile_eps_mean = np.empty(len(ov_points)+1, dtype=float)
        tile_eps_mean_err = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_data = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_enf_data_err = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_enf_data_max = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_data_max_err = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_data_min = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_data_min_err = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_data_mean = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_data_mean_err = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_gp = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_enf_gp_err = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_enf_gp_max = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_gp_max_err = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_gp_min = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_gp_min_err = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_gp_mean = np.empty(len(ov_points)+1, dtype=float)
        tile_enf_gp_mean_err = np.empty(len(ov_points)+1, dtype=float)
        tile_res_data = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_res_data_max = np.empty(len(ov_points)+1, dtype=float)
        tile_res_data_min = np.empty(len(ov_points)+1, dtype=float)
        tile_res_data_mean = np.empty(len(ov_points)+1, dtype=float)
        tile_res_gp = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_res_gp_err = np.empty((len(ov_points)+1, 16), dtype=float)
        tile_res_gp_max = np.empty(len(ov_points)+1, dtype=float)
        tile_res_gp_max_err = np.empty(len(ov_points)+1, dtype=float)
        tile_res_gp_min = np.empty(len(ov_points)+1, dtype=float)
        tile_res_gp_min_err = np.empty(len(ov_points)+1, dtype=float)
        tile_res_gp_mean = np.empty(len(ov_points)+1, dtype=float)
        tile_res_gp_mean_err = np.empty(len(ov_points)+1, dtype=float)
        for ch in range(1, 17):
            amp_gain = amp_gain_df.loc[(amp_gain_df['pcb_pos'] == pos_val + 1) & (amp_gain_df['ch'] == ch), 'amp_gain']
            amp_gain = amp_gain.iloc[0]
            filter_ch_str = f"ch == {ch}"
            ch_df = run_df.Filter(filter_ch_str)
            ov = ch_df.AsNumpy(["ov"])["ov"]
            vol = ch_df.AsNumpy(["vol"])["vol"]
            ov_err = ch_df.AsNumpy(["vbd_err"])["vbd_err"]
            ref_mu = ch_df.AsNumpy(["ref_mu"])["ref_mu"]
            ref_mu_err = ch_df.AsNumpy(["ref_mu_err"])["ref_mu_err"]
            ref_pde = reff_pdes[pos_val] 
            mu = ch_df.AsNumpy(["mu"])["mu"]
            mu_err = ch_df.AsNumpy(["mu_err"])["mu_err"]
            ap = ch_df.AsNumpy(["ap"])["ap"]
            ap_err =  ch_df.AsNumpy(["ap_err"])["ap_err"]
            enf_gp = ch_df.AsNumpy(["enf_GP"])["enf_GP"]
            enf_gp_err =  ch_df.AsNumpy(["enf_GP_err"])["enf_GP_err"]
            enf_data = ch_df.AsNumpy(["enf_data"])["enf_data"]
            enf_data_err =  ch_df.AsNumpy(["enf_data_err"])["enf_data_err"]
            res_data = ch_df.AsNumpy(["res_data"])["res_data"]
            res_gp = ch_df.AsNumpy(["res_GP"])["res_GP"]
            res_gp_err =  ch_df.AsNumpy(["res_GP_err"])["res_GP_err"]
            gain = ch_df.AsNumpy(["gain"])["gain"]
            gain_err =  ch_df.AsNumpy(["gain_err"])["gain_err"]
            for i in range(len(ap)):
                if ov[i] < 5.:
                    ap[i] == 0.02
                    ap_err[i] = 0.01
            dcr = ch_df.AsNumpy(["dcr"])["dcr"]
            dcr_err = ch_df.AsNumpy(["dcr_err"])["dcr_err"]
            for i in range(len(dcr)):
                if dcr[i] == 0:
                    dcr[i] += 1
            lambda_ = ch_df.AsNumpy(["lambda"])["lambda"]
            lambda_err = ch_df.AsNumpy(["lambda_err"])["lambda_err"]
            # Zip the three lists together based on vol
            zipped_lists = zip(vol, ov, mu, mu_err, ap, ap_err, gain, gain_err, dcr, dcr_err,
                               lambda_, lambda_err, enf_gp, enf_gp_err, enf_data, enf_data_err,
                               res_gp, res_gp_err, res_data)
            
            # Sort the zipped lists based on vol (which is the first element in each tuple)
            sorted_lists = sorted(zipped_lists, key=lambda x: x[0])
            
            # Unzip the sorted lists
            vol, ov, mu, mu_err, ap, ap_err, gain, gain_err, dcr, dcr_err, lambda_, lambda_err, enf_gp, enf_gp_err, enf_data, enf_data_err, res_gp, res_gp_err, res_data = zip(*sorted_lists)
            
            # Define the finer vol values to interpolate
            vol_fine = np.arange(1, 6.01, 0.01)
            n_element = len(vol_fine) 
            nearest_indices = [np.abs(np.array(vol) - v).argmin() for v in vol_fine]
            # Interpolate ov based on vol_fine
            ov = np.interp(vol_fine, vol, ov)
            ref_mu = np.resize(ref_mu, n_element)
            ref_mu_err = np.resize(ref_mu_err, n_element)
            # For the errors, we'll find the nearest original vol point for each interpolated vol
            # mu
            spl_mu = InterpolatedUnivariateSpline(vol, mu)
            mu = spl_mu(vol_fine)
            mu_err = np.array([weighted_error(v, vol, mu_err) for v in vol_fine])
            # lambda
            spl_lambda = InterpolatedUnivariateSpline(vol, lambda_)
            lambda_ = spl_lambda(vol_fine)
            lambda_err = np.array([weighted_error(v, vol, lambda_err) for v in vol_fine])
            # ap
            spl_ap = InterpolatedUnivariateSpline(vol, ap)
            ap = spl_ap(vol_fine)
            ap_err = np.array([weighted_error(v, vol, ap_err) for v in vol_fine])
            # gain
            spl_gain = InterpolatedUnivariateSpline(vol, gain)
            gain = spl_gain(vol_fine) / amp_gain / 1.602e-19 * 1e-12
            gain_err = np.array([weighted_error(v, vol, gain_err) for v in vol_fine]) / amp_gain / 1.602e-19 * 1e-12
            # dcr
            spl_dcr = InterpolatedUnivariateSpline(vol, dcr)
            dcr = spl_dcr(vol_fine)
            dcr_err = np.array([weighted_error(v, vol, dcr_err) for v in vol_fine])
            # enf_data
            spl_enf_data = InterpolatedUnivariateSpline(vol, enf_data)
            enf_data = spl_enf_data(vol_fine)
            enf_data_err = np.array([weighted_error(v, vol, enf_data_err) for v in vol_fine])
            # enf_gp
            spl_enf_gp = InterpolatedUnivariateSpline(vol, enf_gp)
            enf_gp = spl_enf_gp(vol_fine)
            enf_gp_err = np.array([weighted_error(v, vol, enf_gp_err) for v in vol_fine])
            # res_gp
            spl_res_gp = InterpolatedUnivariateSpline(vol, res_gp)
            res_gp = spl_res_gp(vol_fine)
            res_gp_err = np.array([weighted_error(v, vol, res_gp_err) for v in vol_fine])
            # res_data
            spl_res_data = InterpolatedUnivariateSpline(vol, res_data)
            res_data = spl_res_data(vol_fine)
            

            pct = 1 - np.exp(-lambda_)
            pct_err = lambda_err 
            pde_abs = mu * ref_pde / ref_mu / 1.04
            pde_abs_err = pde_abs * np.sqrt((mu_err / mu)**2 + (0.01)**2 +(ref_mu_err / ref_mu)**2)
            Pcn = pct + ap # currently AP is set to be 1%
            Pcn_err = np.sqrt(pct_err**2 + ap_err**2)
            pde_eff = 0.51 + 0.35 * Pcn + 0.84 * Pcn**2 + (4.2*1e-4 + 2*1e-4* Pcn) * dcr
            # Error for 0.35 * Pcn
            delta_35Pcn = 0.35 * Pcn_err
            
            # Error for 0.84 * Pcn^2
            delta_84Pcn2 = 0.84 * 2 * Pcn * Pcn_err
            
            # Error for (4.2e-4 + 2e-4 * Pcn) * dcr
            term_error = np.sqrt((4.2e-4 * dcr_err)**2 + (2e-4 * dcr * Pcn_err)**2)
            
            # Total error for pde_eff
            pde_eff_err = np.sqrt(delta_35Pcn**2 + delta_84Pcn2**2 + term_error**2)
            delta_eps = pde_abs * 0.896 /0.9 - pde_eff
            # Error for pde_abs * 0.896 / 0.9
            x = pde_abs * 0.896 / 0.9
            delta_x = x * (pde_abs_err / pde_abs)
            
            # Placeholder for error in pde_eff. You'll need to provide or calculate this.
            delta_y = pde_eff_err  # This should be defined elsewhere in your code based on the formula for pde_eff
            
            # Final error for delta_eps
            delta_eps_err = np.sqrt(delta_x**2 + delta_y**2)
            index_max = np.argmax(delta_eps)
            #index_3p5V = nearest_index(ov, 3.5)
            #index_4V = nearest_index(ov, 4)
            #index_4p5V = nearest_index(ov, 4.5)
            index_ov_points = np.empty(len(ov_points), dtype=int) #= nearest_index(ov, ov_points)
            for i in range(len(ov_points)):
                index_ov_point = nearest_index(ov, ov_points[i])
                index_ov_points[i] = index_ov_point
            tile_vop[ch-1] = ov[index_max]
            for i in range(len(ov_points)+1):
                if i == len(ov_points):
                    tile_pde[i][ch-1] = pde_abs[index_max]
                    tile_pde_err[i][ch-1] = pde_abs_err[index_max]
                    tile_dcr[i][ch-1] = dcr[index_max]
                    tile_dcr_err[i][ch-1] = dcr_err[index_max]
                    tile_pct[i][ch-1] = pct[index_max]
                    tile_pct_err[i][ch-1] = pct_err[index_max]
                    tile_gain[i][ch-1] = gain[index_max]
                    tile_gain_err[i][ch-1] = gain_err[index_max]
                    tile_eps[i][ch-1] = delta_eps[index_max]
                    tile_eps_err[i][ch-1] = delta_eps_err[index_max]
                    tile_enf_data[i][ch-1] = enf_data[index_max]
                    tile_enf_data_err[i][ch-1] = enf_data_err[index_max]
                    tile_enf_gp[i][ch-1] = enf_gp[index_max]
                    tile_enf_gp_err[i][ch-1] = enf_gp_err[index_max]
                    tile_res_gp[i][ch-1] = res_gp[index_max]
                    tile_res_gp_err[i][ch-1] = res_gp_err[index_max]
                    tile_res_data[i][ch-1] = res_data[index_max]
                else:
                    tile_pde[i][ch-1] = pde_abs[index_ov_points[i]]
                    tile_pde_err[i][ch-1] = pde_abs_err[index_ov_points[i]]
                    tile_dcr[i][ch-1] = dcr[index_ov_points[i]]
                    tile_dcr_err[i][ch-1] = dcr_err[index_ov_points[i]]
                    tile_pct[i][ch-1] = pct[index_ov_points[i]]
                    tile_pct_err[i][ch-1] = pct_err[index_ov_points[i]]
                    tile_gain[i][ch-1] = gain[index_ov_points[i]]
                    tile_gain_err[i][ch-1] = gain_err[index_ov_points[i]]
                    tile_eps[i][ch-1] = delta_eps[index_ov_points[i]]
                    tile_eps_err[i][ch-1] = delta_eps_err[index_ov_points[i]]
                    tile_enf_data[i][ch-1] = enf_data[index_ov_points[i]]
                    tile_enf_data_err[i][ch-1] = enf_data_err[index_ov_points[i]]
                    tile_enf_gp[i][ch-1] = enf_gp[index_ov_points[i]]
                    tile_enf_gp_err[i][ch-1] = enf_gp_err[index_ov_points[i]]
                    tile_res_gp[i][ch-1] = res_gp[index_ov_points[i]]
                    tile_res_gp_err[i][ch-1] = res_gp_err[index_ov_points[i]]
                    tile_res_data[i][ch-1] = res_data[index_ov_points[i]]
                
        tile_vop_max = np.max(tile_vop)
        tile_vop_min = np.min(tile_vop)
        tile_vop_mean = np.mean(tile_vop)
        for i in range(len(ov_points)+1):
            tile_pde_max[i] = np.max(tile_pde[i])
            tile_pde_min[i] = np.min(tile_pde[i])
            tile_pde_mean[i] = np.mean(tile_pde[i])
            tile_pde_max_err[i] = tile_pde_err[i][np.argmax(tile_pde[i])]
            tile_pde_min_err[i] = tile_pde_err[i][np.argmin(tile_pde[i])]
            tile_pde_mean_err[i] = np.sqrt(np.sum(tile_pde_err[i]**2) / len(tile_pde_err[i]))
            tile_dcr_max[i] = np.max(tile_dcr[i])
            tile_dcr_min[i] = np.min(tile_dcr[i])
            tile_dcr_mean[i] = np.mean(tile_dcr[i])
            tile_dcr_max_err[i] = tile_dcr_err[i][np.argmax(tile_dcr[i])]
            tile_dcr_min_err[i] = tile_dcr_err[i][np.argmin(tile_dcr[i])]
            tile_dcr_mean_err[i] = np.sqrt(np.sum(tile_dcr_err[i]**2) / len(tile_dcr_err[i]))
            tile_pct_max[i] = np.max(tile_pct[i])
            tile_pct_min[i] = np.min(tile_pct[i])
            tile_pct_mean[i] = np.mean(tile_pct[i])
            tile_pct_max_err[i] = tile_pct_err[i][np.argmax(tile_pct[i])]
            tile_pct_min_err[i] = tile_pct_err[i][np.argmin(tile_pct[i])]
            tile_pct_mean_err[i] = np.sqrt(np.sum(tile_pct_err[i]**2) / len(tile_pct_err[i]))
            tile_gain_max[i] = np.max(tile_gain[i])
            tile_gain_min[i] = np.min(tile_gain[i])
            tile_gain_mean[i] = np.mean(tile_gain[i])
            tile_gain_max_err[i] = tile_gain_err[i][np.argmax(tile_gain[i])]
            tile_gain_min_err[i] = tile_gain_err[i][np.argmin(tile_gain[i])]
            tile_gain_mean_err[i] = np.sqrt(np.sum(tile_gain_err[i]**2) / len(tile_gain_err[i]))
            tile_eps_max[i] = np.max(tile_eps[i])
            tile_eps_min[i] = np.min(tile_eps[i])
            tile_eps_mean[i] = np.mean(tile_eps[i])
            tile_eps_max_err[i] = tile_eps_err[i][np.argmax(tile_eps[i])]
            tile_eps_min_err[i] = tile_eps_err[i][np.argmin(tile_eps[i])]
            tile_eps_mean_err[i] = np.sqrt(np.sum(tile_eps_err[i]**2) / len(tile_eps_err[i]))
            tile_enf_data_max[i] = np.max(tile_enf_data[i])
            tile_enf_data_min[i] = np.min(tile_enf_data[i])
            tile_enf_data_mean[i] = np.mean(tile_enf_data[i])
            tile_enf_data_max_err[i] = tile_enf_data_err[i][np.argmax(tile_enf_data[i])]
            tile_enf_data_min_err[i] = tile_enf_data_err[i][np.argmin(tile_enf_data[i])]
            tile_enf_data_mean_err[i] = np.sqrt(np.sum(tile_enf_data_err[i]**2) / len(tile_enf_data_err[i]))
            tile_enf_gp_max[i] = np.max(tile_enf_gp[i])
            tile_enf_gp_min[i] = np.min(tile_enf_gp[i])
            tile_enf_gp_mean[i] = np.mean(tile_enf_gp[i])
            tile_enf_gp_max_err[i] = tile_enf_gp_err[i][np.argmax(tile_enf_gp[i])]
            tile_enf_gp_min_err[i] = tile_enf_gp_err[i][np.argmin(tile_enf_gp[i])]
            tile_enf_gp_mean_err[i] = np.sqrt(np.sum(tile_enf_gp_err[i]**2) / len(tile_enf_gp_err[i]))
            tile_res_gp_max[i] = np.max(tile_res_gp[i])
            tile_res_gp_min[i] = np.min(tile_res_gp[i])
            tile_res_gp_mean[i] = np.mean(tile_res_gp[i])
            tile_res_gp_max_err[i] = tile_res_gp_err[i][np.argmax(tile_res_gp[i])]
            tile_res_gp_min_err[i] = tile_res_gp_err[i][np.argmin(tile_res_gp[i])]
            tile_res_gp_mean_err[i] = np.sqrt(np.sum(tile_res_gp_err[i]**2) / len(tile_res_gp_err[i]))
            tile_res_data_max[i] = np.max(tile_res_data[i])
            tile_res_data_min[i] = np.min(tile_res_data[i])
            tile_res_data_mean[i] = np.mean(tile_res_data[i])

        header = "tsn,run,"
        content = f"{tsn},{run},"
        header += "vbd_diff,vbd_max,vbd_max_err,vbd_min,vbd_min_err,vbd_mean,vbd_mean_err,"
        content+= f"{difference},{max_vbd},{max_vbd_err},{min_vbd},{min_vbd_err},{mean_vbd},{mean_vbd_err},"
        header += "vop_max,vop_min,vop_mean,"
        content+= f"{tile_vop_max},{tile_vop_min},{tile_vop_mean},"
        header += "pde_3v_max,pde_3v_max_err,pde_3v_min,pde_3v_min_err,pde_3v_mean,pde_3v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pde_max[0],tile_pde_max_err[0],tile_pde_min[0],tile_pde_min_err[0],tile_pde_mean[0],tile_pde_mean_err[0])
        header += "pde_3p5v_max,pde_3p5v_max_err,pde_3p5v_min,pde_3p5v_min_err,pde_3p5v_mean,pde_3p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pde_max[1],tile_pde_max_err[1],tile_pde_min[1],tile_pde_min_err[1],tile_pde_mean[1],tile_pde_mean_err[1])
        header += "pde_4v_max,pde_4v_max_err,pde_4v_min,pde_4v_min_err,pde_4v_mean,pde_4v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pde_max[2],tile_pde_max_err[2],tile_pde_min[2],tile_pde_min_err[2],tile_pde_mean[2],tile_pde_mean_err[2])
        header += "pde_4p5v_max,pde_4p5v_max_err,pde_4p5v_min,pde_4p5v_min_err,pde_4p5v_mean,pde_4p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pde_max[3],tile_pde_max_err[3],tile_pde_min[3],tile_pde_min_err[3],tile_pde_mean[3],tile_pde_mean_err[3])
        header += "pde_vop_max,pde_vop_max_err,pde_vop_min,pde_vop_min_err,pde_vop_mean,pde_vop_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pde_max[4],tile_pde_max_err[4],tile_pde_min[4],tile_pde_min_err[4],tile_pde_mean[4],tile_pde_mean_err[4])
        header += "dcr_3v_max,dcr_3v_max_err,dcr_3v_min,dcr_3v_min_err,dcr_3v_mean,dcr_3v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_dcr_max[0],tile_dcr_max_err[0],tile_dcr_min[0],tile_dcr_min_err[0],tile_dcr_mean[0],tile_dcr_mean_err[0])
        header += "dcr_3p5v_max,dcr_3p5v_max_err,dcr_3p5v_min,dcr_3p5v_min_err,dcr_3p5v_mean,dcr_3p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_dcr_max[1],tile_dcr_max_err[1],tile_dcr_min[1],tile_dcr_min_err[1],tile_dcr_mean[1],tile_dcr_mean_err[1])
        header += "dcr_4v_max,dcr_4v_max_err,dcr_4v_min,dcr_4v_min_err,dcr_4v_mean,dcr_4v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_dcr_max[2],tile_dcr_max_err[2],tile_dcr_min[2],tile_dcr_min_err[2],tile_dcr_mean[2],tile_dcr_mean_err[2])
        header += "dcr_4p5v_max,dcr_4p5v_max_err,dcr_4p5v_min,dcr_4p5v_min_err,dcr_4p5v_mean,dcr_4p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_dcr_max[3],tile_dcr_max_err[3],tile_dcr_min[3],tile_dcr_min_err[3],tile_dcr_mean[3],tile_dcr_mean_err[3])
        header += "dcr_vop_max,dcr_vop_max_err,dcr_vop_min,dcr_vop_min_err,dcr_vop_mean,dcr_vop_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_dcr_max[4],tile_dcr_max_err[4],tile_dcr_min[4],tile_dcr_min_err[4],tile_dcr_mean[4],tile_dcr_mean_err[4])
        header += "pct_3v_max,pct_3v_max_err,pct_3v_min,pct_3v_min_err,pct_3v_mean,pct_3v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pct_max[0],tile_pct_max_err[0],tile_pct_min[0],tile_pct_min_err[0],tile_pct_mean[0],tile_pct_mean_err[0])
        header += "pct_3p5v_max,pct_3p5v_max_err,pct_3p5v_min,pct_3p5v_min_err,pct_3p5v_mean,pct_3p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pct_max[1],tile_pct_max_err[1],tile_pct_min[1],tile_pct_min_err[1],tile_pct_mean[1],tile_pct_mean_err[1])
        header += "pct_4v_max,pct_4v_max_err,pct_4v_min,pct_4v_min_err,pct_4v_mean,pct_4v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pct_max[2],tile_pct_max_err[2],tile_pct_min[2],tile_pct_min_err[2],tile_pct_mean[2],tile_pct_mean_err[2])
        header += "pct_4p5v_max,pct_4p5v_max_err,pct_4p5v_min,pct_4p5v_min_err,pct_4p5v_mean,pct_4p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pct_max[3],tile_pct_max_err[3],tile_pct_min[3],tile_pct_min_err[3],tile_pct_mean[3],tile_pct_mean_err[3])
        header += "pct_vop_max,pct_vop_max_err,pct_vop_min,pct_vop_min_err,pct_vop_mean,pct_vop_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_pct_max[4],tile_pct_max_err[4],tile_pct_min[4],tile_pct_min_err[4],tile_pct_mean[4],tile_pct_mean_err[4])
        header += "gain_3v_max,gain_3v_max_err,gain_3v_min,gain_3v_min_err,gain_3v_mean,gain_3v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_gain_max[0],tile_gain_max_err[0],tile_gain_min[0],tile_gain_min_err[0],tile_gain_mean[0],tile_gain_mean_err[0])
        header += "gain_3p5v_max,gain_3p5v_max_err,gain_3p5v_min,gain_3p5v_min_err,gain_3p5v_mean,gain_3p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_gain_max[1],tile_gain_max_err[1],tile_gain_min[1],tile_gain_min_err[1],tile_gain_mean[1],tile_gain_mean_err[1])
        header += "gain_4v_max,gain_4v_max_err,gain_4v_min,gain_4v_min_err,gain_4v_mean,gain_4v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_gain_max[2],tile_gain_max_err[2],tile_gain_min[2],tile_gain_min_err[2],tile_gain_mean[2],tile_gain_mean_err[2])
        header += "gain_4p5v_max,gain_4p5v_max_err,gain_4p5v_min,gain_4p5v_min_err,gain_4p5v_mean,gain_4p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_gain_max[3],tile_gain_max_err[3],tile_gain_min[3],tile_gain_min_err[3],tile_gain_mean[3],tile_gain_mean_err[3])
        header += "gain_vop_max,gain_vop_max_err,gain_vop_min,gain_vop_min_err,gain_vop_mean,gain_vop_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_gain_max[4],tile_gain_max_err[4],tile_gain_min[4],tile_gain_min_err[4],tile_gain_mean[4],tile_gain_mean_err[4])
        header += "enf_gp_3v_max,enf_gp_3v_max_err,enf_gp_3v_min,enf_gp_3v_min_err,enf_gp_3v_mean,enf_gp_3v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_gp_max[0],tile_enf_gp_max_err[0],tile_enf_gp_min[0],tile_enf_gp_min_err[0],tile_enf_gp_mean[0],tile_enf_gp_mean_err[0])
        header += "enf_gp_3p5v_max,enf_gp_3p5v_max_err,enf_gp_3p5v_min,enf_gp_3p5v_min_err,enf_gp_3p5v_mean,enf_gp_3p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_gp_max[1],tile_enf_gp_max_err[1],tile_enf_gp_min[1],tile_enf_gp_min_err[1],tile_enf_gp_mean[1],tile_enf_gp_mean_err[1])
        header += "enf_gp_4v_max,enf_gp_4v_max_err,enf_gp_4v_min,enf_gp_4v_min_err,enf_gp_4v_mean,enf_gp_4v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_gp_max[2],tile_enf_gp_max_err[2],tile_enf_gp_min[2],tile_enf_gp_min_err[2],tile_enf_gp_mean[2],tile_enf_gp_mean_err[2])
        header += "enf_gp_4p5v_max,enf_gp_4p5v_max_err,enf_gp_4p5v_min,enf_gp_4p5v_min_err,enf_gp_4p5v_mean,enf_gp_4p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_gp_max[3],tile_enf_gp_max_err[3],tile_enf_gp_min[3],tile_enf_gp_min_err[3],tile_enf_gp_mean[3],tile_enf_gp_mean_err[3])
        header += "enf_gp_vop_max,enf_gp_vop_max_err,enf_gp_vop_min,enf_gp_vop_min_err,enf_gp_vop_mean,enf_gp_vop_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_gp_max[4],tile_enf_gp_max_err[4],tile_enf_gp_min[4],tile_enf_gp_min_err[4],tile_enf_gp_mean[4],tile_enf_gp_mean_err[4])
        header += "enf_data_3v_max,enf_data_3v_max_err,enf_data_3v_min,enf_data_3v_min_err,enf_data_3v_mean,enf_data_3v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_data_max[0],tile_enf_data_max_err[0],tile_enf_data_min[0],tile_enf_data_min_err[0],tile_enf_data_mean[0],tile_enf_data_mean_err[0])
        header += "enf_data_3p5v_max,enf_data_3p5v_max_err,enf_data_3p5v_min,enf_data_3p5v_min_err,enf_data_3p5v_mean,enf_data_3p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_data_max[1],tile_enf_data_max_err[1],tile_enf_data_min[1],tile_enf_data_min_err[1],tile_enf_data_mean[1],tile_enf_data_mean_err[1])
        header += "enf_data_4v_max,enf_data_4v_max_err,enf_data_4v_min,enf_data_4v_min_err,enf_data_4v_mean,enf_data_4v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_data_max[2],tile_enf_data_max_err[2],tile_enf_data_min[2],tile_enf_data_min_err[2],tile_enf_data_mean[2],tile_enf_data_mean_err[2])
        header += "enf_data_4p5v_max,enf_data_4p5v_max_err,enf_data_4p5v_min,enf_data_4p5v_min_err,enf_data_4p5v_mean,enf_data_4p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_data_max[3],tile_enf_data_max_err[3],tile_enf_data_min[3],tile_enf_data_min_err[3],tile_enf_data_mean[3],tile_enf_data_mean_err[3])
        header += "enf_data_vop_max,enf_data_vop_max_err,enf_data_vop_min,enf_data_vop_min_err,enf_data_vop_mean,enf_data_vop_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_enf_data_max[4],tile_enf_data_max_err[4],tile_enf_data_min[4],tile_enf_data_min_err[4],tile_enf_data_mean[4],tile_enf_data_mean_err[4])
        header += "eps_3v_max,eps_3v_max_err,eps_3v_min,eps_3v_min_err,eps_3v_mean,eps_3v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_eps_max[0],tile_eps_max_err[0],tile_eps_min[0],tile_eps_min_err[0],tile_eps_mean[0],tile_eps_mean_err[0])
        header += "eps_3p5v_max,eps_3p5v_max_err,eps_3p5v_min,eps_3p5v_min_err,eps_3p5v_mean,eps_3p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_eps_max[1],tile_eps_max_err[1],tile_eps_min[1],tile_eps_min_err[1],tile_eps_mean[1],tile_eps_mean_err[1])
        header += "eps_4v_max,eps_4v_max_err,eps_4v_min,eps_4v_min_err,eps_4v_mean,eps_4v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_eps_max[2],tile_eps_max_err[2],tile_eps_min[2],tile_eps_min_err[2],tile_eps_mean[2],tile_eps_mean_err[2])
        header += "eps_4p5v_max,eps_4p5v_max_err,eps_4p5v_min,eps_4p5v_min_err,eps_4p5v_mean,eps_4p5v_mean_err,"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},".format(tile_eps_max[3],tile_eps_max_err[3],tile_eps_min[3],tile_eps_min_err[3],tile_eps_mean[3],tile_eps_mean_err[3])
        header += "eps_vop_max,eps_vop_max_err,eps_vop_min,eps_vop_min_err,eps_vop_mean,eps_vop_mean_err"
        content+= "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}".format(tile_eps_max[4],tile_eps_max_err[4],tile_eps_min[4],tile_eps_min_err[4],tile_eps_mean[4],tile_eps_mean_err[4])
        header += "\n"
        content+= "\n"
        print(header)
        print(content)
        if tsn_err_flag:
            tsn = -1
        with open(f"parameter_{tsn}.csv","a") as file:
            if write_header:
                file.write(header)
                write_header = False
            file.write(content)
        
            

            


