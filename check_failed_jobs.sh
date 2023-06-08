#!/bin/bash

# Check if the threshold argument is provided
if [[ -z $1 ]]; then
  echo "Usage: ./check_failed_jobs.sh <threshold_in_mb>"
  exit 1
fi

# Get the threshold from the input argument
threshold_mb=$1

# Get the list of files
files=$(ls -l | awk '{print $9}')
# Define the threshold in MB

# Array to store file names without the ".root" suffix
files_to_resubmit=()

# Iterate over the files
for file in $files; do
  # Check if the file is smaller than a certain threshold (e.g., 1 MB)
  if [[ -f $file ]]  && [[ $file == *.root ]]; then
    # Get the file size in MB
    size_mb=$(du -m "$file" | cut -f1)

    # Compare the file size with the threshold
    if (( size_mb < threshold_mb ))
    then
        echo "Failed job: $file (Size: ${size_mb}MB)"
        # Remove the ".root" suffix and add the file name to the array
        files_to_resubmit+=("${file%.root}")
    fi
    # Add your code here to resubmit the job or perform any other actions
  fi
done

# Prompt the user to resubmit jobs
read -p "Ready to resubmit jobs? (y/n): " choice
if [[ $choice == "y" ]]
then
    # Loop through the array and resubmit jobs
    for file in "${files_to_resubmit[@]}"
    do
        # Construct the job script file name
        job_script="jobs/$file.sh"
        echo "Resubmitting job: $job_script"
        
        # Run the hep_sub command with the job script file
        hep_sub "$job_script"
    done
else
    echo "Jobs not resubmitted."
fi
