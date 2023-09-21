import os, sys
import numpy as np
import pandas as pd

decoder_csv = os.path.abspath(sys.argv[1])
matcher_csv = os.path.abspath(sys.argv[2])
if len(sys.argv) == 4:
  output_path = os.path.abspath(sys.argv[3])
else:
  output_path = os.getcwd()
name_short = matcher_csv.split("/")[-2]
components = name_short.split("_")
runNumber = int(components[2])
runType = components[0]
filename = f"positions_{runType}_run_{runNumber}"
csv_output = output_path + "/" + filename + ".csv"
tex_output = output_path + "/" + filename + ".tex"


if not os.path.isdir(output_path):
    os.makedirs(output_path)

def find_nearest(value, array):
    """Find the nearest value in an array to the given value."""
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def format_new_x(value, exp_x):
    if abs(value - exp_x) > 3:
        return "\\textcolor{red}{" + "{:0.2f}".format(value) + "}"
    else:
        return "{:0.2f}".format(value)

def format_new_y(value, exp_y):
    if abs(value - exp_y) > 3:
        return "\\textcolor{red}{" + "{:0.2f}".format(value) + "}"
    else:
        return "{:0.2f}".format(value)

# Define a function to format each row for LaTeX
def format_row(row):
    formatted_row = row.copy()
    for col in ['new_x', 'new_y']:
        exp_col = 'exp_' + col.split('_')[-1]
        est_col = col.split('_')[-1]
        decoder_col = 'decoder_' + col.split('_')[-1]
        if abs(row[col] - row[est_col]) < 0.001 and row[col] != row[decoder_col]:
            formatted_row[col] = "\\textcolor{green}{" + "{:0.2f}".format(row[col]) + "}"
        elif abs(row[col] - row[exp_col]) > 3:
            formatted_row[col] = "\\textcolor{red}{" + "{:0.2f}".format(row[col]) + "}"
            formatted_row[exp_col] = "\\textcolor{blue}{" + "{:0.2f}".format(row[exp_col]) + "}"
        else:
            formatted_row[col] = "{:0.2f}".format(row[col])
    return formatted_row


# Reading the first CSV file
df1 = pd.read_csv(decoder_csv)

if runType == "main":
    df1.at[0, 'run'] = int(runNumber)
# Read the CSV file into a pandas DataFrame
if not os.path.isdir(matcher_csv):
    df2 = pd.read_csv(matcher_csv)
else:
    all_data = []
    for filename in os.listdir(matcher_csv):
        if filename.endswith(".csv"):
            file_path = os.path.join(matcher_csv, filename)
            data = pd.read_csv(file_path)
            all_data.append(data)
    #df2 = pd.concat(all_data, ignore_index=True)
    if len(all_data) == 0:
        exit()
    df2 = pd.concat(all_data, ignore_index=True)

try:
    df2 = df2.sort_values(('point'), ascending=True).reset_index(drop=True)
except KeyError:
    print("Warning: No 'point' column found in 'df' for sorting.")

# Round all numerical columns in df1 to one decimal place
df1 = df1.round(3)

# Round all numerical columns in df2 to one decimal place
df2 = df2.round(3)
# Sort df2 by the 'point' column in ascending order
df2 = df2.sort_values(by='point', ascending=True)
# Sort df1 by the 'point' column in ascending order
df1 = df1.sort_values(by='point', ascending=True)

# Merge asymmetric errors into x and y columns for df2
#df2['est_x'] = df2['x'].astype(str) + " [" + df2['x_left_error'].astype(str) + "," + df2['x_right_error'].astype(str) + "]"
#df2['est_y'] = df2['y'].astype(str) + " [" + df2['y_up_error'].astype(str) + "," + df2['y_down_error'].astype(str) + "]"

# Drop the unnecessary columns
df2.drop(columns=['run','var','chi2','x_left_error', 'x_right_error', 'y_up_error', 'y_down_error'], inplace=True)

# Checking if both dataframes have the same number of rows
#if len(df1) != len(df2):
#    print("Warning: The two dataframes have different numbers of rows.")

# Combining the two dataframes by columns, adding keys to distinguish the source of each column
#combined_df = pd.concat([df1, df2], axis=1)
# Merging the two dataframes based on the 'point' column
combined_df = pd.merge(df1, df2, on='point', how='inner') # You can also use 'outer' or 'left' or 'right' based on your requirement

# Initialize new columns for new_x and new_y with NaN values
combined_df['new_x'] = None
combined_df['new_y'] = None

# Loop through each row in the DataFrame
for index, row in combined_df.iterrows():
    x = row['x']
    y = row['y']
    decoder_x = row['decoder_x']
    decoder_y = row['decoder_y']
    exp_x = row['exp_x']
    exp_y = row['exp_y']
    is_last_point = index == len(combined_df) - 1
    
    if abs(decoder_x - x) < 3:
        new_x = decoder_x
    elif abs(exp_x - x) < 3:
        new_x = exp_x
    elif index > 0:  # Not the first row
        prev_decoder_x = combined_df.loc[index - 1, 'decoder_x']
        prev_exp_x = combined_df.loc[index - 1, 'exp_x']
        
        if abs(prev_decoder_x - x) < 3:
            new_x = prev_decoder_x
        elif abs(prev_exp_x - x) < 3:
            new_x = prev_exp_x
        else:
            if not is_last_point:  # Assuming you have a way to determine if it's the last point
                next_decoder_x = combined_df.loc[index + 1, 'decoder_x']
                next_exp_x = combined_df.loc[index + 1, 'exp_x']
                if abs(next_decoder_x - x) < 3:
                    new_x = next_decoder_x
                elif abs(next_exp_x - x) < 3:
                    new_x = next_exp_x
                #elif abs(prev_decoder_x - next_decoder_x) < 0.1 and prev_decoder_x < 999: # in case next = prev = 1000
                #    new_x = prev_decoder_x
    else:
        new_x = x
    
    if abs(decoder_y - y) < 3:
        new_y = decoder_y
    elif abs(exp_y - y) < 3:
        new_y = exp_y
    elif index > 0:  # Not the first row
        prev_decoder_y = combined_df.loc[index - 1, 'decoder_y']
        prev_exp_y = combined_df.loc[index - 1, 'exp_y']
        
        if abs(prev_decoder_y - y) < 3:
            new_y = prev_decoder_y
        elif abs(prev_exp_y - y) < 3:
            new_y = prev_exp_y
        else:
            if not is_last_point:  # Assuming you have a way to determine if it's the last point
                next_decoder_y = combined_df.loc[index + 1, 'decoder_y']
                next_exp_y = combined_df.loc[index + 1, 'exp_y']
                if abs(next_decoder_y - y) < 3:
                    new_y = next_decoder_y
                elif abs(next_exp_y - y) < 3:
                    new_y = next_exp_y
                #elif abs(prev_decoder_y - next_decoder_y) < 0.1 and prev_decoder_y < 1000:
                #    new_y = prev_decoder_y
    else:
        new_y = y

    if runType == "light":
        new_x_grid = [0, 3.07, 9.07, 15.07, 21.07, 27.07, 33.07, 39.07, 45.07, 51.07, 57.07, 63.07, 69.07]
        new_y_grid = [0, -6.0, -12.0, -18.0, -24.0, -30.0, -36.0, -42.0, -48.0, -54.0, -60.0, -66.0]
        
        # ... (The rest of your code to calculate new_x and new_y)
        
        # Snap new_x and new_y to the nearest grid values
        new_x = find_nearest(new_x, new_x_grid)
        new_y = find_nearest(new_y, new_y_grid)
    elif runType == "main":
        if x < 6 and x > 2:
            new_x = 0.
        elif x >= 6:
            new_x = 1000
        else:
            new_x = -1000
        if y < 0 and y > -4:
            new_y = 0.
        elif y >= 0:
            new_y = 1000.
        else:
            new_y = -1000.

    # Update new_x and new_y columns with calculated values
    combined_df.loc[index, 'new_x'] = new_x
    combined_df.loc[index, 'new_y'] = new_y

# Save the updated DataFrame to a new CSV file
combined_df.to_csv(csv_output, index=False)


# Split the combined dataframe into two halves
n = len(combined_df)
first_half_df = combined_df.iloc[:n//2]
second_half_df = combined_df.iloc[n//2:]

# Reorder the columns
ordered_cols = ['run', 'point', 'decoder_x', 'x', 'exp_x', 'new_x', 'decoder_y', 'y', 'exp_y', 'new_y']
first_half_df = first_half_df[ordered_cols]
second_half_df = second_half_df[ordered_cols]
first_half_df = first_half_df.round(2) 
second_half_df = second_half_df.round(2)
# Apply the format_row function to each row of the first_half_df and second_half_df
first_half_df_formatted = first_half_df.apply(format_row, axis=1)
second_half_df_formatted = second_half_df.apply(format_row, axis=1)

# Convert these formatted DataFrames to LaTeX
latex_code1 = first_half_df_formatted.to_latex(index=False, escape=False, multirow=True, float_format="{:0.2f}".format)
latex_code2 = second_half_df_formatted.to_latex(index=False, escape=False, multirow=True, float_format="{:0.2f}".format)

headers = '''\\documentclass{article}
\\usepackage{graphicx} % Required for inserting images
\\usepackage{booktabs}
\\usepackage{pdflscape}
\\usepackage{color}  % <-- Add this line
\\title{light-match}
\\author{cobby319126 }
\\date{September 2023}
\\begin{document}
\\begin{landscape}
'''
end = '''
\\end{landscape}
\\maketitle
\\section{Introduction}
\\end{document}
'''
print_output = False
if print_output:
    print(headers, "\n", sep="")
    print(latex_code1, sep="")
    print(latex_code2, sep="")
    print(end, "\n", sep="")

# Open the .tex file for writing
with open(tex_output, 'w') as f:
    # Write the LaTeX headers
    f.write(headers + "\n")
    
    # Write the LaTeX code for the first half
    f.write(latex_code1 + "\n")
    
    # Write the LaTeX code for the second half
    f.write(latex_code2 + "\n")
    
    # Write the LaTeX end content
    f.write(end + "\n")
