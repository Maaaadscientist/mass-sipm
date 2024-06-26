import numpy as np
import pandas as pd
import math
import os
import time
import scipy.stats as stats
import matplotlib.pyplot as plt
import ROOT
import argparse  # Import argparse module

# Constants
CHANNELS_PER_TILE = 16  # Number of channels per SiPM tile
#N = 20  # Number of tests
#ov = 3.0
PHOTONS_PER_TEST = 9846  # Number of photons simulated in each test

single_pe_resolution = 0.15
# Example parameters
#dcr = 50.
PTE = 0.836  # Probability of photon hitting SiPM active area 815074 / 975251
keys = [round(2.5 + 0.1 * i, 1) for i in range(46)]

p_ap_list = [0., 0., 0., 0.0023121856076775654, 0.0018802001816537676, 0.004592260188738675, 0.005603494363346231, 0.0066575250176804214, 0.007667411782256014, 0.008433497021108608, 0.00843078391998287, 0.008385036021587812, 0.007904183469901863, 0.008850636694014601, 0.008713131143955609, 0.008717673859595676, 0.007030457136347763, 0.004947314444334691, 0.0057115686477610976, 0.007579198898779196, 0.009598977237236433, 0.010580060518829364, 0.012581271931600347, 0.013350082161514336, 0.014199856115907178, 0.015615447989311698, 0.016500705177208246, 0.0169721013382356, 0.017793987388636398, 0.018500698254727343, 0.01892180184767479, 0.019482180748078306, 0.0207172660202462, 0.02106353453046995, 0.021925275947001836, 0.023004779571738715, 0.023665011882735572, 0.024921758712276504, 0.026189829123050276, 0.027538547025168413, 0.028165703300553683, 0.028201262809919716, 0.030093832832628876, 0.031007762004127252, 0.03217048726827821, 0.03363866384548797]
# Pairing the keys with values from p_ap_list
pap_ov_dict = dict(zip(keys, p_ap_list))
#optical_crosstalk_param = 0.2  # Parameter for generalized Poisson distribution
pde_dict = {"max": 0.44, "typical":0.47}
pct_dict = {"max": 0.15, "typical":0.12}
dcr_dict = {"max": 41.7, "typical":13.9}
pap_dict = {"max": 0.08, "typical":0.04}
gain_dict = {"max": 1e6, "typical":4e6}


def parse_arguments():
    parser = argparse.ArgumentParser(description='Photon Simulation Parameters')
    parser.add_argument('--N', type=int, default=20, help='Number of tests')
    parser.add_argument('--energy', type=float, default=1.0, help='deposit energy (MeV)')
    parser.add_argument('--ov', type=float, default=3.0, help='Overvoltage, for HPK test: -1: max, -2: typical')
    parser.add_argument('--level', type=int, required=True, help='Random level: 1: whole detector, 2: by tile, 3: by channel, 4: by channel and different ov ')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, default='all_hist.root', help='Output ROOT file path')
    parser.add_argument('--seed', type=int,default=123456, help='Random seed for simulation', required=False)
    args = parser.parse_args()
    return args.N, args.energy, args.ov, args.level, args.input, args.output, args.seed

# Parse command line arguments
N, energy, ov, level, input_file, output_file, seed = parse_arguments()
# Initialize random seed
if seed is not None:
    np.random.seed(seed)

# Load the CSV file
df = pd.read_csv(input_file)

# Create a reference dictionary
reference_dict = {(row['tsn'], row['ch']): idx for idx, row in df.iterrows()}
# Get unique values from the 'tsn' column
unique_tsn_values = df['tsn'].unique()

def clear_screen():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For macOS and Linux (here, os.name is 'posix')
    else:
        _ = os.system('clear')

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

def borel_pmf(k, lambda_):
    return (lambda_ * k)**(k - 1) * np.exp(-k * lambda_) / math.factorial(k)


def generate_random_borel(lambda_, max_k=20):
    # Generate PMF values
    pmf_values = [borel_pmf(k+1, lambda_) for k in range(max_k)]

    # Normalize the PMF
    total = sum(pmf_values)
    normalized_pmf = [value / total for value in pmf_values]

    # Random sampling based on the PMF
    return np.random.choice(range(max_k), p=normalized_pmf)

def simulate_photon(ov, level=3):
    """ Simulate the journey of a single photon. """
    # Check if photon hits the active area
    if np.random.rand() > PTE:
        return  0, 0, 0, 0, 0

    # Select a random tile and channel
    tile = np.random.choice(unique_tsn_values)
    vbd_average = average_value_from_csv(df, "vbd", {"tsn":tile})
    channel = np.random.randint(CHANNELS_PER_TILE) + 1

    # Check if photon is detected
    vbd_ch = get_value_by_tsn_ch(df, reference_dict, tile, channel, "vbd")
    ch_ov = ov - vbd_ch + vbd_average
    if ch_ov < 2.5:
        ch_ov = 2.5
    if ch_ov > 7.0:
        ch_ov = 7.0
    if level == 2:
        pde = average_value_from_csv(df, "pde", {"tsn":tile})
    elif level == 3:
        pde = get_value_by_tsn_ch(df, reference_dict, tile, channel, f"pde_{ov:.1f}")
    elif level == 4:
        pde = get_value_by_tsn_ch(df, reference_dict, tile, channel, f"pde_{ch_ov:.1f}")

    if pde >= 1:
        pde = 0.5
    elif pde<=0:
        pde = 0.01
    detected = np.random.rand() < pde
    if not detected:
        return  1, 0, 0, 0, 0
    # Simulate optical crosstalk
    
    #pct = get_value_by_tsn_ch(df, reference_dict, tile, channel, pct_name)
    if level == 2:
        pct = average_value_from_csv(df, "pct", {"tsn":tile})
    elif level == 3:
        pct = get_value_by_tsn_ch(df, reference_dict, tile, channel, f"pct_{ov:.1f}")
    elif level == 4:
        pct = get_value_by_tsn_ch(df, reference_dict, tile, channel, f"pct_{ch_ov:.1f}")

    if pct >= 1:
        pct = 0.5
    elif pct <= 0:
        pct = 0.01
    lambda_ = np.log(1 / (1 - pct))
    electrons = 1 + generate_random_borel(lambda_)

    # Simulate afterpulsing
    p_ap = pap_ov_dict[round(ov,1)]
    afterpulses = np.sum(np.random.rand(electrons) < p_ap)
    
    if level == 2:
        gain = average_value_from_csv(df, "gain", {"tsn":tile}) / 1e6
    elif level == 3:
        gain = get_value_by_tsn_ch(df, reference_dict, tile, channel, f"gain_{ov:.1f}") / 1e6
    elif level == 4:
        gain = get_value_by_tsn_ch(df, reference_dict, tile, channel, f"gain_{ch_ov:.1f}") / 1e6

    if gain <= 0:
        gain = average_value_from_csv(df, "gain", {"tsn":tile}) / 1e6
    gain = 1.
    gain += np.random.normal(0, gain*single_pe_resolution)
    return 1, 1, electrons * (1-lambda_), electrons * (1-lambda_) + afterpulses - electrons * p_ap, gain * ( electrons * (1-lambda_) + afterpulses - electrons * p_ap)

def simulate_dcr(ov):
    ov_str = str(round(ov,1))
    pde_name = "pde_" + ov_str
    pct_name = "pct_" + ov_str
    dcr_name = "dcr_" + ov_str
    gain_name = "gain_" + ov_str

    if ov == -1:
        dcr = dcr_dict["max"]
        gain = gain_dict["max"]/1e6
        pct = pct_dict["max"]
    elif ov == -2:
        dcr = dcr_dict["typical"]
        gain = gain_dict["typical"]/1e6
        pct = pct_dict["typical"]
    else:
        dcr = average_value_from_csv(df, dcr_name)
        gain = average_value_from_csv(df, gain_name)/1e6
        pct = average_value_from_csv(df, pct_name)
    gain = 1
    init_pe = dcr * 10 * 1e6 * 300 * 1e-9
    pe = np.random.poisson(init_pe)
    pe_ct = pe
    lambda_ = np.log(1 / (1 - pct))
    for _ in range(int(pe)):
        pe_ct += generate_random_borel(lambda_)

    # Simulate afterpulsing
    
    if ov == -1:
        p_ap = pap_dict["max"]
    elif ov == -2:
        p_ap = pap_dict["typical"]
    else:
        p_ap = pap_ov_dict[round(ov,1)]
    pe_ap = np.sum(np.random.rand(pe_ct) < p_ap)
    gain += np.random.normal(0, gain*single_pe_resolution)
    return init_pe, pe, pe_ct * (1-lambda_), pe_ct* (1-lambda_) + pe_ap - pe_ct * p_ap, ( pe_ct* (1-lambda_) + pe_ap - pe_ct * p_ap) * gain, gain
        
def main():
    init_photons = int(PHOTONS_PER_TEST * energy)
    Nbins = init_photons*2
    # Data collection arrays
    h_init = ROOT.TH1F("hist_init", "hist_init", Nbins,0,Nbins)
    h_LS = ROOT.TH1F("hist_LS", "hist_LS", Nbins,0,Nbins)
    h_pte = ROOT.TH1F("hist_PTE", "hist_PTE", Nbins,0,Nbins)
    h_pde = ROOT.TH1F("hist_PDE", "hist_PDE", Nbins,0,Nbins)
    h_ct = ROOT.TH1F("hist_ct", "hist_ct", Nbins,0,Nbins)
    h_ap = ROOT.TH1F("hist_ap", "hist_ap", Nbins,0,Nbins)
    h_dcr = ROOT.TH1F("hist_dcr", "hist_dcr", Nbins,0,Nbins)
    h_charge = ROOT.TH1F("hist_charge", "hist_charge", Nbins,0,Nbins)
    hist_dcr_init = ROOT.TH1F("dcr_init", "dcr_init", 2000,0,2000)
    hist_dcr_poisson = ROOT.TH1F("dcr_poisson", "dcr_poisson", 2000,0,2000)
    hist_dcr_ct = ROOT.TH1F("dcr_ct", "dcr_ct", 2000,0,2000)
    hist_dcr_ap = ROOT.TH1F("dcr_ap", "dcr_ap", 2000,0,2000)
    if ov == -1 or ov == -2:
        random_level = 1
    else:
        random_level = level
    start_time = time.time()
    for i in range(N):
        
        # Refresh output every X events (e.g., every 10 events)
        if (i + 1) % 10 == 0 or i == N - 1:
            elapsed_time = time.time() - start_time
            percent_complete = ((i + 1) / N) * 100
            avg_time_per_event = elapsed_time / (i + 1)
            estimated_total_time = avg_time_per_event * N
            estimated_remaining_time = estimated_total_time - elapsed_time

            clear_screen()
            print(f"Event {i + 1}/{N} complete")
            print(f"Completion: {percent_complete:.2f}%")
            print(f"Elapsed Time: {elapsed_time:.2f} seconds")
            print(f"Estimated Total Time: {estimated_total_time:.2f} seconds")
            print(f"Estimated Time Remaining: {estimated_remaining_time:.2f} seconds")
            time.sleep(0.1)  # Small delay to ensure the screen refresh is noticeable

        LS_photons = np.random.poisson(init_photons)
        dcr_results = simulate_dcr(ov) 
        dcr_init = dcr_results[0]
        dcr_poisson_pe = dcr_results[1]
        dcr_pe_ct  =dcr_results[2] 
        dcr_pe_ctap= dcr_results[3]
        dcr_charge=dcr_results[4]
        dcr_gain=dcr_results[5]
        hist_dcr_init.Fill(dcr_init)
        hist_dcr_poisson.Fill(dcr_poisson_pe)
        hist_dcr_ct.Fill(dcr_pe_ct)
        hist_dcr_ap.Fill(dcr_pe_ctap)
        if random_level == 2 or random_level == 3 or random_level ==4:
            test_results = [simulate_photon(ov, random_level) for _ in range(LS_photons)]
            hit_photons = sum(r[0] for r in test_results)
            detected_photons = sum(r[1] for r in test_results)
            detected_electrons = sum(r[2] for r in test_results)
            total_electrons = sum(r[3] for r in test_results)
            signal_charge = sum(r[4] for r in test_results)
            dcr_pe = detected_photons + dcr_poisson_pe - dcr_init
            ct_pe = detected_electrons + dcr_pe_ct - dcr_init 
            ap_pe = total_electrons + dcr_pe_ctap - dcr_init
            total_charge = signal_charge + dcr_charge - dcr_init*dcr_gain
            h_LS.Fill(LS_photons)
            h_pte.Fill(hit_photons)
            h_pde.Fill(detected_photons)
            h_dcr.Fill(dcr_pe)
            h_ct.Fill(ct_pe)
            h_ap.Fill(ap_pe)
            h_charge.Fill(total_charge)
        elif random_level == 1:
            hit_photons = np.random.binomial(LS_photons, PTE) 
            if ov == -1:
                PDE = pde_dict["max"]
            elif ov == -2:
                PDE = pde_dict["typical"]
            else:
                PDE = average_value_from_csv(df, f"pde_{ov:.1f}") 
            detected_photons = np.random.binomial(hit_photons, PDE)
            if ov == -1:
                PCT = pct_dict["max"]
            elif ov == -2:
                PCT = pct_dict["typical"]
            else:
                PCT = average_value_from_csv(df, f"pct_{ov:.1f}") 
            lambda_ = np.log(1 / (1 - PCT))
            dcr_pe = detected_photons + dcr_poisson_pe - dcr_init
            ct_pe = dcr_pe
            for i in range(int(dcr_pe)):
                extra_pe = generate_random_borel(lambda_)
                ct_pe += extra_pe
            ct_pe *= 1 - lambda_
            ap_pe = ct_pe
            if ov == -1:
                p_ap = pap_dict["max"]
            elif ov == -2:
                p_ap = pap_dict["typical"]
            else:
                p_ap = pap_ov_dict[round(ov,1)]
            ap_pe += np.random.binomial(ct_pe, p_ap)
            ap_pe -= ct_pe * p_ap
            if ov == -1:
                GAIN =  gain_dict["max"]/1e6
            elif ov == -2:
                GAIN = gain_dict["typical"]/1e6
            else:
                GAIN = average_value_from_csv(df, f"gain_{ov:.1f}") / 1e6
            GAIN = 1.
            total_charge = 0
            for _ in range(int(ap_pe)):
                single_GAIN = GAIN + np.random.normal(0, GAIN*single_pe_resolution)
                total_charge += single_GAIN
            
        
        h_init.Fill(PHOTONS_PER_TEST * energy)
        h_LS.Fill(LS_photons)
        h_pte.Fill(hit_photons)
        h_pde.Fill(detected_photons)
        h_dcr.Fill(dcr_pe)
        h_ct.Fill(ct_pe)
        h_ap.Fill(ap_pe)
        h_charge.Fill(total_charge)
    
    # At the end, where the ROOT file is saved
    f1 = ROOT.TFile(output_file, "recreate")
    h_init.Write()
    h_LS.Write()
    h_pte.Write()
    h_pde.Write()
    h_dcr.Write()
    h_ct.Write()
    h_ap.Write()
    h_charge.Write()
    hist_dcr_init.Write()
    hist_dcr_poisson.Write()
    hist_dcr_ct.Write()
    hist_dcr_ap.Write()
    f1.Close()

if __name__ == "__main__":
    main()
