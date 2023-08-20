#!/bin/bash

# Check if the threshold argument and directory argument are provided
if [[ -z $1 ]]; then
  echo "Usage: ./doGeoFit.sh <input_path>"
  exit 1
fi
# Get the directory location from the input argument and resolve to absolute path
input_path=$(realpath "$1")
directory="/junofs/users/wanghanwen/sipm-massive"
cd $directory
sleep 3
. ./env_lcg.sh
python=$(which python3)
# Construct and execute the command
harvest_command="$python script/design-coordinates.py ${input_path}"
echo "Executing command: ${harvest_command}"
$harvest_command

cd -
