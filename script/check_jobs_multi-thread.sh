#!/bin/bash

# Check if the threshold argument and directory argument are provided
if [[ -z $1 || -z $2 ]]; then
  echo "Usage: ./check_jobs_multi-thread.sh <threshold_in_KB> <directory> [interactive_mode]"
  exit 1
fi

# Get the threshold from the input argument
threshold_kb=$1

# Get the directory location from the input argument and resolve to absolute path
directory=$(realpath "$2")

# Get the interactive mode option (true or false)
interactive_mode=${3:-true}

file_type=$4

resubmit_mode=${5:-false}
# Get the list of "*.sh" files in the "jobs" directory
script_files=$(find $directory/jobs -type f -name "*.sh" -printf "%f\n")

# Get the list of "*.root" files in the specified directory that pass the size threshold
root_files=$(find $directory/$file_type -maxdepth 1 -type f -name "*.$file_type"  -printf "%f\n")

# Array to store script files to be resubmitted
scripts_to_resubmit=()

# Iterate over the script files
for script_file in $script_files; do
  # Remove the ".sh" extension to get the corresponding root file name
  root_file="${script_file%.sh}.$file_type"
  
  # Check if the corresponding root file exists
  if [[ -f "$directory/$file_type/$root_file" ]]; then
    # File exists, remove it from the list of root files
    root_files=$(echo "$root_files" | grep -v "$root_file")
    #size_kb=$(du -k "$directory/$file_type/$root_file" | awk '{print $1}')
    size_bytes=$(du -b "$directory/$file_type/$root_file" | awk '{print $1}')

    #echo $size_bytes
    # Compare the file size with the threshold
    if (( size_bytes <= threshold_kb ))
    then
        #echo "Failed job: $root_file (Size: ${size_bytes}KB)"
        # Remove the ".root" suffix and add the file name to the array
        scripts_to_resubmit+=("$script_file")
    fi
  else
    # Corresponding root file does not exist, add the script file to the array
    scripts_to_resubmit+=("$script_file")
  fi
done

# Check if interactive mode is enabled
if [[ $interactive_mode == true ]]; then
  # Prompt the user to resubmit jobs
  for a_script_file in "${scripts_to_resubmit[@]}"; do
    echo "$a_script_file"
  done
  length=${#scripts_to_resubmit[@]}
  echo -e "${GREEN}Number of jobs to be submitted (resubmitted): $length ${NC}"
  # Prompt the user to resubmit jobs
  echo "Ready to submit jobs? (Y/N):"
  read choice
  
  # Directory to be added to PATH
  htcondor_dir="/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin"
  
  
  # Check if the user wants to resubmit jobs
  if [[ $choice == "y" ]]; then
    echo "will submit $length jobs."
    # Check if the directory is already in the PATH
    if echo "$PATH" | grep -q "$htcondor_dir"; then
      echo "Directory already exists in PATH. No changes needed."
    else
      # Add the directory to the PATH variable
      export PATH="$htcondor_dir:$PATH"
      echo "Directory added to PATH."
    fi
    # Loop through the array and resubmit jobs
    for script_file in "${scripts_to_resubmit[@]}"; do
      echo "Resubmitting job: $script_file"
      hep_sub "$directory/jobs/$script_file" -o /dev/null -e /dev/null -cpu 4 -mem 4000
    done
  else
    echo "Jobs not resubmitted."
  fi
else
  
  #echo "resubmit all jobs? (Y: resubmit all, N: resubmit failed ones, other: do nothing)"
  #read choice
  if [[ $resubmit_mode == false ]]; then
    length=${#script_files[@]}
    echo -e "${GREEN}Number of jobs to be submitted (resubmitted): $length ${NC}"
    # Interactive mode is disabled, skip printing information and resubmitting jobs
    echo "Interactive mode disabled. Skipping printing information and asking confirmation."
    # Directory to be added to PATH
    htcondor_dir="/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin"
    
    # Check if the directory is already in the PATH
    if ! echo "$PATH" | grep -q "$htcondor_dir"; then
      # Add the directory to the PATH variable
      export PATH="$htcondor_dir:$PATH"
      echo "Directory added to PATH."
    fi
    # Loop through the array and resubmit jobs
    for one_script_file in $script_files; do
      echo "Resubmitting job: $one_script_file"
      hep_sub "$directory/jobs/$one_script_file" -o /dev/null -e /dev/null -cpu 4 -mem 4000
    done
  fi

  if [[ $resubmit_mode == true ]]; then
    # Prompt the user to resubmit jobs
    for a_script_file in "${scripts_to_resubmit[@]}"; do
      echo "$a_script_file"
    done
    length=${#scripts_to_resubmit[@]}
    echo -e "${GREEN}Number of jobs to be submitted (resubmitted): $length ${NC}"
    # Interactive mode is disabled, skip printing information and resubmitting jobs
    echo "Interactive mode disabled. Skipping printing information and asking confirmation."
    # Directory to be added to PATH
    htcondor_dir="/afs/ihep.ac.cn/soft/common/sysgroup/hep_job/bin"
    
    length=${#scripts_to_resubmit[@]}
    # Check if the directory is already in the PATH
    if ! echo "$PATH" | grep -q "$htcondor_dir"; then
      # Add the directory to the PATH variable
      export PATH="$htcondor_dir:$PATH"
      echo "Directory added to PATH."
    fi
    # Loop through the array and resubmit jobs
    for script_file in "${scripts_to_resubmit[@]}"; do
      #echo "Resubmitting job: $script_file"
      hep_sub "$directory/jobs/$script_file" -o /dev/null -e /dev/null -cpu 4 -mem 4000 
    done
  fi
fi
