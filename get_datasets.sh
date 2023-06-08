#!/bin/bash

base_dir="/eos/juno/groups/TAO/taoprod/mass_test/run_data"
output_dir="datasets"

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
    echo "Skipping $directory (already exists)"
  else
    # Use eos find to find files in the current directory and save the results to the output file
    eos find "$dir_path" -type f > "$output_file"
    echo "Generated $output_file"
  fi
done

