#!/bin/bash

base_dir="/eos/juno/groups/TAO/taoprod/mass_test/run_data"
tao_base_dir="/home/shifter/data-hd/transferred_move_dir"
export EOS_MGM_URL="root://junoeos01.ihep.ac.cn"

# Get the list of directories on the TAO server
tao_directories=$(ssh shifter@tao "ls -1 $tao_base_dir")

# Convert the list of TAO directories into an array for easier comparison
readarray -t tao_directory_array <<< "$tao_directories"

# Loop over the TAO directories
for tao_directory in "${tao_directory_array[@]}"; do
  echo "Checking directory: $tao_directory"
  # Check if the entry is a directory (skip files with extensions)
  if [[ "$tao_directory" == *.* ]]; then
    echo "Skipping $tao_directory (not a directory)"
    continue
  fi

  # Fetch file list and sizes from TAO directory in one go, store in a temp file
  ssh shifter@tao "find $tao_base_dir/$tao_directory -type f -exec stat -c '%s %n' {} +" > tao_files.txt

  # Fetch file list and sizes from EOS directory in one go, store in a temp file
  eos find --size $base_dir/$tao_directory -type f > eos_files.txt

  # Now read both files line-by-line and compare
  while read -r tao_line; do
    tao_size=$(echo $tao_line | awk '{print $1}')
    tao_file=$(echo $tao_line | awk '{$1=""; print $0}' | sed 's/^ *//') # Remove leading spaces

    # Strip the common base directory and current sub-directory from the tao_file to make it comparable
    tao_file_short=$(echo "$tao_file" | sed "s~^$tao_base_dir/$tao_directory~~")

    # Search for the transformed tao_file in eos_files.txt
    eos_line=$(grep -F "$tao_file_short" eos_files.txt)
    #echo $tao_file_short
    #eos_line=$(grep -F "$tao_file" eos_files.txt)
    if [[ -n $eos_line ]]; then
      #eos_size=$(echo $eos_line | awk '{print $2}')
      #eos_size=$(echo $eos_line | awk -F= '{print $2}')
      eos_size=$(echo $eos_line | awk '{for(i=1;i<=NF;i++) if ($i ~ /size=/) {split($i,a,"="); print a[2]}}')

      if [ "$tao_size" -eq "$eos_size" ]; then
        :
        #echo "Sizes match for $tao_file: $tao_size bytes."
      else
        echo "Sizes do not match for $tao_file. TAO: $tao_size bytes, EOS: $eos_size bytes."
      fi
    else
      echo "File $tao_file does not exist on EOS."
    fi
  done < tao_files.txt

  # Clean up temp files
  rm -f tao_files.txt eos_files.txt
done

