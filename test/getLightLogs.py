import yaml
import paramiko
from subprocess import Popen, PIPE

password = sys.argv[1]
# Load YAML file
with open('config/light-runs.yaml', 'r') as f:
    data = yaml.safe_load(f)
    light_runs = data['light_runs']

# SSH configuration
hostname = '202.38.128.236'
username = 'tao'
private_key_path = '/afs/ihep.ac.cn/users/w/wanghanwen/.ssh/id_rsa'  # Replace with the path to your SSH key

# Create SSH client
ssh_client = paramiko.SSHClient()
ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # To skip missing host key error, not recommended for production
ssh_client.connect(hostname, username="shifter", key_filename=private_key_path)

# Loop through each light run and execute MySQL command
for run in light_runs:
    mysql_cmd = f'mysql -u tao {password} -h {hostname} -e "SELECT lsid,reff_orn,point_orn,dx,dy FROM tao_mass_tests_data.lscan_data WHERE reff_orn=1 and lsid={run}"'
    
    # Execute command on remote server
    stdin, stdout, stderr = ssh_client.exec_command(mysql_cmd)
    output = stdout.read().decode()
    
    # Save output to text file
    with open(f'datasets/lightLogs/light_run_{run}.txt', 'w') as f:
        f.write(output)

# Close SSH client
ssh_client.close()

