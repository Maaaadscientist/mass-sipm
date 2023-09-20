import os
import pandas as pd

# Read the log file
log_df = pd.read_csv("combined_log.csv")

# Initialize an empty DataFrame to store the final data
final_df = pd.DataFrame(columns=['run_type', 'run_id', 'point', 'new_x', 'new_y', 'total_time'])

# Loop through both directories
for dir_name, run_type in [("positions_light_run", "light"), ("positions_main_run", "main")]:
    for csv_file in os.listdir(dir_name):
        if csv_file.endswith('.csv'):
            # Read the CSV file into a DataFrame
            temp_df = pd.read_csv(os.path.join(dir_name, csv_file))

            # Extract the 'run' id
            run_id = temp_df['run'].iloc[0]

            # Find the corresponding 'total_time' from the log DataFrame
            matching_rows = log_df.loc[(log_df['type'] == run_type) & (log_df['id'] == run_id), 'total_time']

            # Check if matching rows were found
            if not matching_rows.empty:
                total_time = matching_rows.iloc[0]

                # Create a new DataFrame with the needed columns
                new_df = temp_df[['run', 'point', 'new_x', 'new_y']].copy()
                new_df['run_type'] = run_type
                new_df['total_time'] = total_time
                new_df.rename(columns={'run': 'run_id'}, inplace=True)

                # Append to the final DataFrame
                final_df = pd.concat([final_df, new_df])
            else:
                print(f"No matching run_id {run_id} of type {run_type} found in the log.")

# Write the final DataFrame to a new CSV file
final_df.to_csv("combined_data.csv", index=False)

