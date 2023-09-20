import os, sys
import pymysql
import pandas as pd
import paramiko
from sshtunnel import SSHTunnelForwarder

# SSH configurations
ssh_host = '202.38.128.236'
ssh_port = 22
ssh_user = 'shifter'
ssh_private_key = "/afs/ihep.ac.cn/users/w/wanghanwen/.ssh/id_rsa"

# MySQL configurations
mysql_host = '202.38.128.236'
mysql_port = 3306
mysql_user = 'tao'
mysql_password = sys.argv[1]
mysql_db = 'tao_mass_tests_data'

# Create an SSH key object with Paramiko
my_pkey = paramiko.RSAKey(filename=ssh_private_key)

invalid_light_runs = [262, 270]
invalid_main_runs = []
# Function to find rows to drop based on ID

def find_rows_to_drop(df, id_column, type_column, invalid_light_runs, invalid_main_runs):
    indices_to_drop = []
    max_ids = {}  # Dictionary to store the maximum ID for each type
    
    for i, row in df.iterrows():
        row_type = row[type_column]
        row_id = row[id_column]
        
        # Check if the row should be invalidated based on provided lists
        if row_type == 'light' and row_id in invalid_light_runs:
            indices_to_drop.append(i)
            continue
        elif row_type == 'main' and row_id in invalid_main_runs:
            indices_to_drop.append(i)
            continue
        
        # If this type has been encountered before, compare the ID
        if row_type in max_ids:
            if row_id < max_ids[row_type]:
                indices_to_drop.append(i)
            else:
                max_ids[row_type] = max(row_id, max_ids[row_type])
                
        # If encountering this type for the first time, store the ID
        else:
            max_ids[row_type] = row_id
            
    return indices_to_drop


# Create an SSH tunnel
with SSHTunnelForwarder(
    (ssh_host, ssh_port),
    ssh_username=ssh_user,
    ssh_pkey=my_pkey,
    remote_bind_address=(mysql_host, mysql_port)
) as tunnel:

    # Connect to MySQL database
    conn = pymysql.connect(
        host='127.0.0.1',
        port=tunnel.local_bind_port,
        user=mysql_user,
        passwd=mysql_password,
        db=mysql_db
    )

    # Fetch data for light runs
    sql_light = "SELECT lsid, ts FROM lscan_log"
    df_light = pd.read_sql(sql_light, conn)
    df_light['type'] = 'light'

    # Fetch data for main runs
    sql_main = "SELECT sid, ts FROM scan_log"
    df_main = pd.read_sql(sql_main, conn)
    df_main['type'] = 'main'

    # Close MySQL connections
    conn.close()

    # Find indices to drop for each type
    df_main.sort_values(by='ts', inplace=True)
    df_light.sort_values(by='ts', inplace=True)
    indices_to_drop_main = find_rows_to_drop(df_main, 'sid', 'type', invalid_light_runs, invalid_main_runs)
    indices_to_drop_light = find_rows_to_drop(df_light, 'lsid', 'type', invalid_light_runs, invalid_main_runs)

    # Drop those rows
    df_light.drop(indices_to_drop_light, inplace=True)
    df_main.drop(indices_to_drop_main, inplace=True)

    # Combine both DataFrames and rename the columns
    df_combined = pd.concat([df_light.rename(columns={'lsid': 'id'}), df_main.rename(columns={'sid': 'id'})])

    # Sort by timestamp
    df_combined.sort_values(by='ts', inplace=True)
    #indices_to_drop_light = find_rows_to_drop(df_light, 'lsid')

    # Add a column for the signal time length
    df_combined['signal_time_length'] = df_combined['type'].apply(lambda x: 64 if x == 'light' else 192)
    
    # Compute a running sum of signal time for all types of runs
    df_combined['total_time'] = df_combined['signal_time_length'].cumsum() - df_combined['signal_time_length']

    # Write to CSV file or display
    df_combined.to_csv('combined_log.csv', index=False)
    print(df_combined)

