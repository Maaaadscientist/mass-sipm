import os
import paramiko
import argparse
import csv

# Argument parsing
parser = argparse.ArgumentParser(description='Execute script on DAQ machine and fetch results.')
parser.add_argument('-r', '--run', type=int, required=True, help='Run number to execute the script with.')
parser.add_argument('-o', '--output', type=str, default="output.csv", help='Path to save the output CSV file.')
args = parser.parse_args()

# SSH connection details
hostname = "202.38.128.236"
port = 22
username = "shifter"

# Path to your private key
#private_key_path = "/Users/wanghanwen/.ssh/id_rsa"
private_key_path = "/afs/ihep.ac.cn/users/w/wanghanwen/.ssh/id_rsa"

# Create an SSH key instance
mykey = paramiko.RSAKey(filename=private_key_path)  # Use DSSKey, ECDSAKey, or Ed25519Key if not RSA

# Run number
run_number = 333

# Establish an SSH session
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # Automatically adds the server's host key (not safe for production)
ssh.connect(hostname, port, username, pkey=mykey)

# Execute the script on the remote machine
stdin, stdout, stderr = ssh.exec_command(f"/home/shifter/scripts/test.py -n {args.run}")

# Process the output and save to CSV
output = stdout.readlines()

output_dir = "/".join(os.path.abspath(args.output).split("/")[:-1])
if not os.path.isdir(output_dir):
    print("Directory doesn't exist, creating it with path:", output_dir)
    os.makedirs(output_dir)
with open(args.output, 'w', newline='') as csvfile:
    # Modified field names
    writer = csv.DictWriter(csvfile, fieldnames=['run', 'pos', 'batch', 'box', 'tsn'])
    writer.writeheader()
    for line in output:
        line_data = eval(line.strip())
        
        # Drop unnecessary columns
        del line_data['sov']
        del line_data['svolt']
        del line_data['sorn']
        
        # Rename columns and modify values
        line_data['run'] = line_data.pop('sid')
        line_data['pos'] = line_data.pop('torn') - 1

        # Split the tsn value and apply protection logic
        tsn_parts = line_data['tsn'].split('-')
        if len(tsn_parts) == 3:
            batch, box, tsn = tsn_parts
        elif len(tsn_parts) == 2:
            batch, tsn = tsn_parts
            box = '-1'
        elif len(tsn_parts) == 1:
            tsn = tsn_parts[0]
            batch = box = '-1'
        else:
            # Handle unexpected cases, you can modify as needed
            batch = box = tsn = '-1'

        line_data['batch'] = batch
        line_data['box'] = box
        line_data['tsn'] = tsn
        
        writer.writerow(line_data)

# Close the SSH connection
ssh.close()

