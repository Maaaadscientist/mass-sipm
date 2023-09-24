#!/bin/bash

base_dir="/eos/juno/groups/TAO/taoprod/mass_test/run_data"
output_dir="datasets"
export EOS_MGM_URL="root://junoeos01.ihep.ac.cn"

# Get the list of directories
directories=$(eos ls "$base_dir" | sort -r)

# Loop over the directories
for directory in $directories; do
  # Build the full directory path
  dir_path="$base_dir/$directory"
  # Check if the entry is a directory
  if [[ "$directory" == *.* ]]; then
    echo "Skipping $directory (not a directory)"
    continue
  fi

  # Generate the output file path
  output_file="$output_dir/$directory.txt"

  # Check if the output file already exists
  if [[ -f "$output_file" ]]; then
    file_type=""
    if [[ "$output_file" =~ light_run_.* ]]; then
      file_type="light"
    elif [[ "$output_file" =~ main_run_.* ]]; then
      file_type="main"
    fi
    
    count_lines=$(grep -c ".data$" "$output_file")

    if [[ "$file_type" == "main" && $count_lines -lt 192 ]] || [[ "$file_type" == "light" && $count_lines -lt 64 ]]; then
      eos find "$dir_path" -type f > "$output_file"
      echo "Regenerated $output_file due to insufficient .data lines"
    else
      echo "Skipping $directory (already exists)"
    fi
  else
    eos find "$dir_path" -type f > "$output_file"
    echo "Generated $output_file"
  fi
 # # Check if the output file already exists
 # if [[ -f "$output_file" ]]; then
 #   echo "Skipping $directory (already exists)"
 # else
 #   # Use eos find to find files in the current directory and save the results to the output file
 #   eos find "$dir_path" -type f > "$output_file"
 #   echo "Generated $output_file"
 # fi
done

