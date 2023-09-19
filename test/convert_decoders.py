import os, sys
import pandas as pd

if len(sys.argv) < 3:
    raise IOError("not enough input arguments")

text_input = os.path.abspath(sys.argv[1])
csv_input = os.path.abspath(sys.argv[2])
if len(sys.argv) == 4:
    csv_output = os.path.abspath(sys.argv[3])
else:
    csv_output = os.getcwd() + "/merged.csv"

DEF_TSTAGE_X_MAX = 56590        # max value of x-axis
DEF_TSTAGE_Y_MAX = 54140        # max value of y-axis

CONVERSION_FACTOR = 819.1952077093372  # the factor by which to divide x and y

def convert_coordinates(row):
    x, y = row['dx'], row['dy']
    converted_x, converted_y = 1000, 1000
    
    # Check if x and y are negative
    if x < 0 or y < 0:
        converted_x = 1000
        converted_y = 1000
        
    # Check if x and y exceed their respective maximums
    elif x > DEF_TSTAGE_X_MAX or y > DEF_TSTAGE_Y_MAX:
        converted_x = 1000
        converted_y = 1000
        
    # Convert x and y by dividing them by the conversion factor
    else:
        converted_x = x / CONVERSION_FACTOR
        converted_y = y / CONVERSION_FACTOR - 66.07
        
    return pd.Series([converted_x, converted_y], index=['decoder_x', 'decoder_y'])

# Read the tab-separated text file into a pandas DataFrame
df = pd.read_csv(text_input, sep='\t')

# Rename the 'lsid' column to 'run' and other renames
df.rename(columns={'lsid': 'run', 'point_orn':'point'}, inplace=True)


## Apply the function
#df[['decoder_x', 'decoder_y']] = df.apply(convert_coordinates, axis=1)
#df = df.round(3)
## Drop the unnecessary columns
##df.drop(columns=['reff_orn', 'dx', 'dy'], inplace=True)
## Write the DataFrame to a CSV file
#df.to_csv('output.csv', index=False)
## Read the expected.csv file into another DataFrame
df2 = pd.read_csv(csv_input)

# Merge the two DataFrames on 'run' and 'point'
merged_df = pd.merge(df, df2[[ 'point','exp_x', 'exp_y']], on=['point'], how='left')

merged_df[['decoder_x', 'decoder_y']] = merged_df.apply(convert_coordinates, axis=1)
# Drop the unnecessary columns
merged_df.drop(columns=['reff_orn', 'dx', 'dy'], inplace=True)

merged_df = merged_df.round(3)
# Save the DataFrame to a new CSV file
merged_df.to_csv(csv_output, index=False)

