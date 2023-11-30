import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import ROOT
import argparse  # Import argparse module

# Constants
CHANNELS_PER_TILE = 16  # Number of channels per SiPM tile
#N = 20  # Number of tests
#ov = 3.0
PHOTONS_PER_TEST = 9846  # Number of photons simulated in each test

# Example parameters
#dcr = 50.
PTE = 0.836  # Probability of photon hitting SiPM active area 815074 / 975251
keys = [round(2.5 + 0.1 * i, 1) for i in range(46)]

p_ap_list = [0., 0., 0., 0.0023121856076775654, 0.0018802001816537676, 0.004592260188738675, 0.005603494363346231, 0.0066575250176804214, 0.007667411782256014, 0.008433497021108608, 0.00843078391998287, 0.008385036021587812, 0.007904183469901863, 0.008850636694014601, 0.008713131143955609, 0.008717673859595676, 0.007030457136347763, 0.004947314444334691, 0.0057115686477610976, 0.007579198898779196, 0.009598977237236433, 0.010580060518829364, 0.012581271931600347, 0.013350082161514336, 0.014199856115907178, 0.015615447989311698, 0.016500705177208246, 0.0169721013382356, 0.017793987388636398, 0.018500698254727343, 0.01892180184767479, 0.019482180748078306, 0.0207172660202462, 0.02106353453046995, 0.021925275947001836, 0.023004779571738715, 0.023665011882735572, 0.024921758712276504, 0.026189829123050276, 0.027538547025168413, 0.028165703300553683, 0.028201262809919716, 0.030093832832628876, 0.031007762004127252, 0.03217048726827821, 0.03363866384548797]
# Pairing the keys with values from p_ap_list
pap_dict = dict(zip(keys, p_ap_list))
#optical_crosstalk_param = 0.2  # Parameter for generalized Poisson distribution

def parse_arguments():
    parser = argparse.ArgumentParser(description='Photon Simulation Parameters')
    parser.add_argument('--N', type=int, default=20, help='Number of tests')
    parser.add_argument('--ov', type=float, default=3.0, help='Overvoltage')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='all_hist.root', help='Output ROOT file path')
    args = parser.parse_args()
    return args.N, args.ov, args.input, args.output

# Parse command line arguments
N, ov, input_file, output_file = parse_arguments()

# Load the CSV file
df = pd.read_csv(input_file)

# Create a reference dictionary
reference_dict = {(row['tsn'], row['ch']): idx for idx, row in df.iterrows()}
# Get unique values from the 'tsn' column
unique_tsn_values = df['tsn'].unique()

def average_value_from_csv(df, column_name, selection_criteria=None):
    # Check if selection criteria are provided
    if selection_criteria:
        # Apply the selection criteria
        for key, value in selection_criteria.items():
            df = df[df[key] == value]

    # Calculate and return the average value of the specified column
    if not df.empty:
        return df[column_name].mean()
    else:
        return "No matching data or empty column"

# Function to get a specific value from a row based on tsn, ch, and column name
def get_value_by_tsn_ch(df, reference_dict, tsn, ch, column_name):
    row_index = reference_dict.get((tsn, ch))
    if row_index is not None and column_name in df.columns:
        return df.at[row_index, column_name]
    else:
        return "Value not found"

def generalized_poisson_pmf(k, mu, lambda_):
    exp_term = np.exp(-(mu + k * lambda_))
    main_term = mu * ((mu + k * lambda_) ** (k - 1))
    factorial_term = math.factorial(k)
    return (main_term * exp_term) / factorial_term

def generate_random_generalized_poisson(mu, lambda_, max_k=20):
    # Generate PMF values
    pmf_values = [generalized_poisson_pmf(k, mu, lambda_) for k in range(max_k)]

    # Normalize the PMF
    total = sum(pmf_values)
    normalized_pmf = [value / total for value in pmf_values]

    # Random sampling based on the PMF
    return np.random.choice(range(max_k), p=normalized_pmf)

def simulate_photon(ov):
    """ Simulate the journey of a single photon. """
    # Check if photon hits the active area
    if np.random.rand() > PTE:
        return 0, 0, 0, 0, 0

    # Select a random tile and channel
    tile = np.random.choice(unique_tsn_values)
    vbd_average = average_value_from_csv(df, "vbd", {"tsn":tile})
    channel = np.random.randint(CHANNELS_PER_TILE) + 1

    # Check if photon is detected
    vbd_ch = get_value_by_tsn_ch(df, reference_dict, tile, channel, "vbd")
    ov = ov - vbd_ch + vbd_average
    if ov < 2.5:
        ov = 2.5
    if ov > 7.0:
        ov = 7.0
    ov_str = str(round(ov,1))
    pde_name = "pde_" + ov_str
    pct_name = "pct_" + ov_str
    gain_name = "gain_" + ov_str
    pde = get_value_by_tsn_ch(df, reference_dict, tile, channel, pde_name)
    if pde >= 1:
        pde = 0.5
    elif pde<0:
        pde = 0
    detected = np.random.rand() <= pde
    if not detected:
        return 1, 0, 0, 0, 0

    # Simulate optical crosstalk
    
    pct = get_value_by_tsn_ch(df, reference_dict, tile, channel, pct_name)
    if pct >= 1:
        pct = 0.5
    elif pct <= 0:
        pct = 0
    lambda_ = np.log(1 / (1 - pct))
    electrons = generate_random_generalized_poisson(1, lambda_)

    # Simulate afterpulsing
    p_ap = pap_dict[round(ov,1)]
    afterpulses = np.sum(np.random.rand(electrons) < p_ap)
    
    gain = get_value_by_tsn_ch(df, reference_dict, tile, channel, gain_name) / 1e6

    return 1, detected, electrons, electrons + afterpulses, gain * (electrons + afterpulses)

def simulate_dcr(ov):
    ov_str = str(round(ov,1))
    pde_name = "pde_" + ov_str
    pct_name = "pct_" + ov_str
    dcr_name = "dcr_" + ov_str
    gain_name = "gain_" + ov_str
    dcr = average_value_from_csv(df, dcr_name)
    gain= average_value_from_csv(df, gain_name) / 1e6
    init_pe = dcr * 10 * 1e6 * 300 * 1e-9
    pe_ct = 0
    pct = np.log(1 / (1 - average_value_from_csv(df, pct_name)))
    for _ in range(int(init_pe)):
        pe_ct += generate_random_generalized_poisson(1, pct)

    # Simulate afterpulsing
    pe_ap = np.sum(np.random.rand(pe_ct) < pap_dict[round(ov,1)])
    return pe_ct + pe_ap, (pe_ct + pe_ap) * gain
        
def main():
    # Data collection arrays
    initial_photons_list = []
    detected_photons_list = []  # List to collect detected photons
    detected_electrons_list = []
    total_electrons_list = []
    dcr_added_electrons_list = []
    # Running N tests
    
    h_init = ROOT.TH1F("hist_init", "hist_init", 20000,0,20000)
    h_LS = ROOT.TH1F("hist_LS", "hist_LS", 20000,0,20000)
    h_pte = ROOT.TH1F("hist_PTE", "hist_PTE", 20000,0,20000)
    h_pde = ROOT.TH1F("hist_PDE", "hist_PDE", 20000,0,20000)
    h_ct = ROOT.TH1F("hist_ct", "hist_ct", 20000,0,20000)
    h_ap = ROOT.TH1F("hist_ap", "hist_ap", 20000,0,20000)
    h_dcr = ROOT.TH1F("hist_dcr", "hist_dcr", 20000,0,20000)
    h_charge = ROOT.TH1F("hist_charge", "hist_charge", 200000,0,200000)
    for _ in range(N):
        LS_photons = np.random.poisson(PHOTONS_PER_TEST)
        test_results = [simulate_photon(ov) for _ in range(LS_photons)]
        hit_photons = sum(r[0] for r in test_results)
        detected_photons = sum(r[1] for r in test_results)
        detected_electrons = sum(r[2] for r in test_results)
        total_electrons = sum(r[3] for r in test_results)
        total_charge = sum(r[4] for r in test_results)
        dcr_electrons = simulate_dcr(ov)[0]
        dcr_charge = simulate_dcr(ov)[1]
        
        h_init.Fill(PHOTONS_PER_TEST)
        h_LS.Fill(LS_photons)
        h_pte.Fill(hit_photons)
        h_pde.Fill(detected_photons)
        h_ct.Fill(detected_electrons)
        h_ap.Fill(total_electrons)
        h_dcr.Fill(total_electrons + dcr_electrons)
        h_charge.Fill(total_charge + dcr_charge)
    
        initial_photons_list.append(hit_photons)
        detected_photons_list.append(detected_photons)
        detected_electrons_list.append(detected_electrons)
        total_electrons_list.append(total_electrons)
        dcr_added_electrons_list.append(total_electrons + dcr_electrons)
    
    # At the end, where the ROOT file is saved
    f1 = ROOT.TFile(output_file, "recreate")
    h_init.Write()
    h_LS.Write()
    h_pte.Write()
    h_pde.Write()
    h_ct.Write()
    h_ap.Write()
    h_dcr.Write()
    h_charge.Write()
    f1.Close()

if __name__ == "__main__":
    main()
## Plotting histograms
#plt.figure(figsize=(20, 5))
#
#plt.subplot(1, 5, 1)
#plt.hist(initial_photons_list, bins=50, color='blue', alpha=0.7)
#plt.title('Initial Photons Detected')
#plt.xlabel('Number of Photons')
#plt.ylabel('Frequency')
#
#plt.subplot(1, 5, 2)
#plt.hist(detected_photons_list, bins=50, color='orange', alpha=0.7)
#plt.title('Detected Photons')
#plt.xlabel('Number of Photons')
#plt.ylabel('Frequency')
#
#plt.subplot(1, 5, 3)
#plt.hist(detected_electrons_list, bins=50, color='green', alpha=0.7)
#plt.title('Detected Electrons After Crosstalk')
#plt.xlabel('Number of Electrons')
#plt.ylabel('Frequency')
#
#plt.subplot(1, 5, 4)
#plt.hist(total_electrons_list, bins=50, color='red', alpha=0.7)
#plt.title('Total Electrons Including Afterpulsing')
#plt.xlabel('Number of Electrons')
#plt.ylabel('Frequency')
#
#plt.subplot(1, 5, 5)
#plt.hist(dcr_added_electrons_list, bins=50, color='deepskyblue', alpha=0.7)
#plt.title('Total Electrons Including DCR')
#plt.xlabel('Number of Electrons')
#plt.ylabel('Frequency')
#plt.tight_layout()
#plt.show()
#
