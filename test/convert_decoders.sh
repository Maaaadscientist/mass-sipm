#!/bin/bash

# Loop over all text files in the 'datasets/lightLogs' directory
for filepath in $(dirname $0)/../datasets/lightLogs/light_run_*.txt; do
  # Extract the run number from the filepath
  filename=$(basename -- "$filepath")
  run_number="${filename##light_run_}"
  run_number="${run_number%.txt}"

  # Construct the output CSV filename
  output_csv=$(dirname $0)"/../datasets/lightRunPositions/light_run_${run_number}.csv"

  # Execute the Python script with appropriate arguments
  cmd="python $(dirname $0)/convert_decoder.py "$filepath" $(dirname $0)/expected.csv "$output_csv""
  echo $cmd
  python $(dirname $0)/convert_decoder.py "$filepath" $(dirname $0)/expected.csv "$output_csv"

  # Print a message indicating progress
  echo "Processed $filepath -> $output_csv"
done

