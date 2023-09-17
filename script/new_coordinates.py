import os, sys
import pandas as pd

decoder_csv = os.path.abspath(sys.argv[1])
matcher_csv = os.path.abspath(sys.argv[2])
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
df2.drop(columns=['run','point','var','chi2','x_left_error', 'x_right_error', 'y_up_error', 'y_down_error'], inplace=True)

# Checking if both dataframes have the same number of rows
if len(df1) != len(df2):
    print("Warning: The two dataframes have different numbers of rows.")

# Combining the two dataframes by columns, adding keys to distinguish the source of each column
combined_df = pd.concat([df1, df2], axis=1)

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
    is_last_point = index == 63
    
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
                elif abs(prev_decoder_x - next_decoder_x) < 0.1 and prev_decoder_x < 999: # in case next = prev = 1000
                    new_x = prev_decoder_x
                else:
                    new_x = x
                    #raise ValueError("No suitable value found for new_x:", f"exp_x:{exp_x}\tx:{x}\tdecoder_x:{decoder_x}")
    else:
        raise ValueError(f"Cannot determine new_x for point {index+1}")
    
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
                elif abs(prev_decoder_y - next_decoder_y) < 0.1 and prev_decoder_y < 1000:
                    new_y = prev_decoder_y
                else:
                    new_y = y
                    #raise ValueError("No suitable value found for new_y")

    else:
        raise ValueError(f"Cannot determine new_y for point {index+1}")

    # Update new_x and new_y columns with calculated values
    combined_df.loc[index, 'new_x'] = new_x
    combined_df.loc[index, 'new_y'] = new_y

# Save the updated DataFrame to a new CSV file
combined_df.to_csv('combined_with_new_columns.csv', index=False)


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

# Convert the second half to LaTeX code
#latex_code2 = second_half_df.to_latex(index=False, multirow=True, float_format="{:0.2f}".format)

## Reorder the columns for the first half DataFrame
#first_half_df = first_half_df[['point', 'decoder_x', 'x', 'exp_x', 'new_x', 'decoder_y', 'y', 'exp_y', 'new_y']]
## Reorder the columns for the second half DataFrame
#second_half_df = second_half_df[['point', 'decoder_x', 'x', 'exp_x', 'new_x', 'decoder_y', 'y', 'exp_y', 'new_y']]
## Create custom formatters for new_x and new_y
#new_x_formatter = lambda x: format_new_x(x, first_half_df['exp_x'])
#new_y_formatter = lambda y: format_new_y(y, first_half_df['exp_y'])
#latex_code1 = first_half_df.to_latex(
#    index=False, 
#    multirow=True, 
#    float_format="{:0.2f}".format,
#    formatters={
#        'new_x': new_x_formatter,
#        'new_y': new_y_formatter
#    }
#)
## Convert the first half to LaTeX code
##latex_code1 = first_half_df.to_latex(index=False, multirow=True, float_format="{:0.2f}".format)

# Optionally, you can save the first LaTeX code to a text file
with open('table1_in_latex.txt', 'w') as f:
    f.write(latex_code1)

# Optionally, you can save the second LaTeX code to a text file
with open('table2_in_latex.txt', 'w') as f:
    f.write(latex_code2)

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
print(headers, "\n", sep="")
print(latex_code1, sep="")
print("\n")
print(latex_code2, sep="")
print(end, "\n", sep="")

