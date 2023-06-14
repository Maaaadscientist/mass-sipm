#!/bin/bash

# Set the default directory to "jobs"
directory="jobs"

# Directory to be added to PATH
htcondor_dir="/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin"

# Check if the directory is already in the PATH
if echo "$PATH" | grep -q "$htcondor_dir"; then
  echo "Directory already exists in PATH. No changes needed."
else
  # Add the directory to the PATH variable
  export PATH="$htcondor_dir:$PATH"
  echo "Directory added to PATH."
fi

# Get the job type argument
job_type="$1"

# If no job type argument provided, set default job types
if [ -z "$job_type" ]; then
  job_type="fit hist"
fi

# Iterate over the files in the directory
for type in $job_type; do
  for file in "$directory"/"$type"*.sh; do
    # Check if the file is a regular file
    if [[ -f "$file" ]]; then
      # Submit the file using hep_sub command
      hep_sub "$file"
    fi
  done
done
