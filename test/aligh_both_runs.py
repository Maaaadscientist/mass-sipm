import os, sys
from sshtunnel import SSHTunnelForwarder
import pymysql
import pandas as pd
import paramiko

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

# Create an SSH tunnel
with SSHTunnelForwarder(
    (ssh_host, ssh_port),
    ssh_username=ssh_user,
    ssh_pkey=my_pkey,
    remote_bind_address=(mysql_host, mysql_port)
) as tunnel:

    # Connect to MySQL database for light runs
    conn_light = pymysql.connect(
        host='127.0.0.1',
        port=tunnel.local_bind_port,
        user=mysql_user,
        passwd=mysql_password,
        db=mysql_db
    )

    # Connect to MySQL database for main runs
    conn_main = pymysql.connect(
        host='127.0.0.1',
        port=tunnel.local_bind_port,
        user=mysql_user,
        passwd=mysql_password,
        db=mysql_db
    )

    # Fetch data for light runs
    sql_light = "SELECT lsid, ts FROM lscan_log"
    df_light = pd.read_sql(sql_light, conn_light)
    df_light['type'] = 'light'

    # Fetch data for main runs
    sql_main = "SELECT sid, ts FROM scan_log"
    df_main = pd.read_sql(sql_main, conn_main)
    df_main['type'] = 'main'

    # Close MySQL connections
    conn_light.close()
    conn_main.close()

    # Combine both DataFrames
    df_combined = pd.concat([df_light.rename(columns={'lsid': 'id'}), df_main.rename(columns={'sid': 'id'})])

    # Sort by timestamp
    df_combined.sort_values(by='ts', inplace=True)

    # Add a column for the signal time length
    df_combined['signal_time_length'] = df_combined['type'].apply(lambda x: 64 if x == 'light' else 192)
    
    # Compute a running sum of signal time for all types of runs
    df_combined['total_time'] = df_combined['signal_time_length'].cumsum() - df_combined['signal_time_length']

    # Write to CSV file or display
    df_combined.to_csv('combined_log.csv', index=False)
    print(df_combined)

