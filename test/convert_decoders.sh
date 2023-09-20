#!/bin/bash

# Loop over all text files in the 'datasets/lightLogs' directory
for filepath in datasets/lightLogs/light_run_*.txt; do
  # Extract the run number from the filepath
  filename=$(basename -- "$filepath")
  run_number="${filename##light_run_}"
  run_number="${run_number%.txt}"

  # Construct the output CSV filename
  output_csv="datasets/lightRunPositions/light_run_${run_number}.csv"

  # Execute the Python script with appropriate arguments
  python test/convert_decoders.py "$filepath" test/expected.csv "$output_csv"

  # Print a message indicating progress
  echo "Processed $filepath -> $output_csv"
done

