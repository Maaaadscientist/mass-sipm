import os, sys
import csv
import re
DEF_TSTAGE_X_MAX = 56590        # max value of x-axis
DEF_TSTAGE_Y_MAX = 54140        # max value of y-axis

CONVERSION_FACTOR = 819.1952077093372  # the factor by which to divide x and y

log_path = os.path.abspath(sys.argv[1])
output_path = os.path.abspath(sys.argv[2])
name_short = log_path.split("/")[-1].replace(".log", "")
components = name_short.split("_")
runNumber = int(components[2])
def convert_coordinates(decoder_x_values, decoder_y_values, expected_x_values, expected_y_values):
    converted_coordinates = []
    
    for x, y, reff_x, reff_y in zip(decoder_x_values, decoder_y_values, expected_x_values, expected_y_values):
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
        
        expected_x = reff_x / CONVERSION_FACTOR
        expected_y = reff_y / CONVERSION_FACTOR - 66.07
        converted_coordinates.append((converted_x, converted_y, expected_x, expected_y))
    
    return converted_coordinates

# Define the regular expression pattern to match lines
# Added '-'? to indicate that numbers can be negative
pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - Current position: # (-?\d+) , Decoder X: (-?\d+), Decoder Y: (-?\d+) Reff\.Position: X: (-?\d+), Y: (-?\d+)')

# Read the log file
with open(log_path, 'r') as f:
    lines = f.readlines()[-800:]

# List to hold lines that match the pattern
matching_lines = []

# Loop through each line to check if it matches the pattern
for line in lines:
    if pattern.match(line):
        matching_lines.append(line)

# Now 'matching_lines' contains all the lines from the log file that match your format.
# Do something with the matching lines

# Compile the regular expression to find relevant data
#pattern_decoder = re.compile(r"Current position: # (\d+) , Decoder X: (-?\d+), Decoder Y: (-?\d+)")

# Regular expression pattern
pattern = re.compile(r"Current position: # (\d+) , Decoder X: (-?\d+), Decoder Y: (-?\d+) Reff.Position: X: (\d+), Y: (\d+)")
## Initialize empty lists to store extracted values
#positions = []
#decoder_x_values = []
#decoder_y_values = []
#
## Iterate through the last 800 lines and apply the regular expression
#for line in matching_lines:
#    match = pattern_decoder.search(line)
#    if match:
#        position, decoder_x, decoder_y = map(int, match.groups())
#        if 1 <= position <= 64:  # Only consider positions between 1 and 64
#            positions.append(position)
#            decoder_x_values.append(decoder_x)
#            decoder_y_values.append(decoder_y)

# Initialize empty lists to store extracted values
positions = []
decoder_x_values = []
decoder_y_values = []
reff_x_values = []
reff_y_values = []

# Extract data from log lines
for line in matching_lines:
    match = pattern.search(line)
    if match:
        position, decoder_x, decoder_y, reff_x, reff_y = map(int, match.groups())
        positions.append(position)
        decoder_x_values.append(decoder_x)
        decoder_y_values.append(decoder_y)
        reff_x_values.append(reff_x)
        reff_y_values.append(reff_y)

# Output the extracted values
converted_coords = convert_coordinates(decoder_x_values, decoder_y_values, reff_x_values, reff_y_values)
print("point,decoder_x, decoder_y, exp_x, exp_y")
for i in range(len(converted_coords)):
    print(i+1, f",{converted_coords[i][0]:.3f}", f",{converted_coords[i][1]:.3f}", f",{converted_coords[i][2]:.3f}", f",{converted_coords[i][3]:.3f}")
with open(output_path + f'/decoder_run{runNumber}.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write the header row
    csv_writer.writerow(['run', 'point', 'decoder_x', 'decoder_y', 'exp_x', 'exp_y'])
    
    # Write the data
    for i, coords in enumerate(converted_coords, 1):
        row = [runNumber, i]
        row.extend([f"{coord:.3f}" for coord in coords])
        csv_writer.writerow(row)
