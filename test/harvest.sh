#!/bin/bash

# Check the number of arguments passed to the script
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <config-file.yaml> <directory-path> <analysis-type>"
  exit 1
fi

# Assign arguments to variables
CONFIG_FILE=$1
ANALYSIS_TYPE=$3
DIRECTORY_PATH="$2/$ANALYSIS_TYPE"

# Output combined CSV file name
COMBINED_CSV="$(pwd)/main_combined.csv"

# Initialize the combined CSV file with header (will be overwritten later)
: > "$COMBINED_CSV"

# Flag for header capture
HEADER_CAPTURED=false

# Check if the given YAML file and directory exist
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file '$CONFIG_FILE' not found."
  exit 1
fi

if [ ! -d "$DIRECTORY_PATH" ]; then
  echo "Directory '$DIRECTORY_PATH' not found."
  exit 1
fi

# Extract the main_runs array from the YAML file using awk and sed
MAIN_RUNS=$(awk '/main_runs:/{flag=1;next}/^[^ ]/{flag=0}flag' $CONFIG_FILE | sed 's/ - //g')

hadd=$(which hadd)
# Loop over the main runs
IFS=$'\n' # Set Internal Field Separator to newline for iteration
for RUN in $MAIN_RUNS; do
  # Zero-pad the run number to form the directory name
  PADDED_RUN=$(printf "%04d" $RUN 2>/dev/null)
  
  if [ $? -ne 0 ]; then
    echo "Failed to process run number: $RUN"
    continue
  fi

  # Construct the directory name
  DIR_NAME="${DIRECTORY_PATH}/main_run_${PADDED_RUN}"

  # Check if the directory exists
  if [ -d "$DIR_NAME" ]; then
    # Execute your commands here
    # For demonstration, I'm just printing the directory name
    echo "Executing commands in $DIR_NAME"
    if [ "$ANALYSIS_TYPE" == "main-reff" ]; then
      COMMAND=".$hadd $DIR_NAME/root/Run${RUN}_Point0.root $DIR_NAME/root/main_*.root"
      echo $COMMAND
      $hadd "$DIR_NAME/root/Run${RUN}_Point0.root" $DIR_NAME/root/main_*.root
      sleep 2 
    elif [ "$ANALYSIS_TYPE" == "main-match" ]; then
      COMMAND=""
      # If header is not yet captured, capture and write it to combined CSV
      if [ "$HEADER_CAPTURED" = false ]; then
        if [ -f "$CSV_FILE" ]; then
          HEADER=$(head -n 1 "$CSV_FILE")
          echo "$HEADER" > "$COMBINED_CSV"
          HEADER_CAPTURED=true
        fi
      fi
      # Combine CSV files
      CSV_FILE="$DIR_NAME/root/Run${RUN}_Point0.csv"
      if [ -f "$CSV_FILE" ]; then
        awk 'NR>1' "$CSV_FILE" >> "$COMBINED_CSV"
      else
        echo "CSV file $CSV_FILE not found, skipping..."
      fi
    fi
    # cd $DIR_NAME
    # Your commands here
  else
    echo "Directory $DIR_NAME not found, skipping..."
  fi
done

