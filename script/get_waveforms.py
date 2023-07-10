import subprocess
import os, sys

inputFile = sys.argv[1]
maxEvents = int(sys.argv[2])
# Define the path to your C++ binary
cpp_binary_path = 'bin/print'

# Define the argument string
argument_string = f'-i {inputFile} -c test.yaml -m {maxEvents}'

# Execute the C++ binary with the argument string and capture its output
process = subprocess.Popen(f'{cpp_binary_path} {argument_string}', shell=True, stdout=subprocess.PIPE)
output, _ = process.communicate()

# Decode the output
output = output.decode('utf-8')

# Print the output
print(output)

