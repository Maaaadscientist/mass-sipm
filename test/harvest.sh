#!/bin/bash

# Check the number of arguments passed to the script
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <config-file.yaml> <directory-path> <analysis-type>"
  exit 1
fi

# Assign arguments to variables
CONFIG_FILE=$1
ANALYSIS_TYPE=$3

# List of allowed analysis types
ALLOWED_ANALYSIS_TYPES=("merge-root" "combine-csv" "main-position" "light-position")
# Declare an associative array for analysis type to suffix mapping
declare -A ANALYSIS_SUFFIXES
ANALYSIS_SUFFIXES=(["merge-root"]="main-reff" ["combine-csv"]="main-match" ["main-position"]="main-match" ["light-position"]="light-match")

# Check if the analysis type is allowed
if [[ ! " ${ALLOWED_ANALYSIS_TYPES[@]} " =~ " ${ANALYSIS_TYPE} " ]]; then
  echo "Error: Invalid analysis type '$ANALYSIS_TYPE'"
  echo "Allowed types are: ${ALLOWED_ANALYSIS_TYPES[@]}"
  exit 1
fi

# Fetch the suffix for the analysis type
SUFFIX=${ANALYSIS_SUFFIXES["$ANALYSIS_TYPE"]}

DIRECTORY_PATH="$(realpath $2)/$SUFFIX"

if [ $ANALYSIS_TYPE == "combine-csv" ]; then
  # Output combined CSV file name
  COMBINED_CSV="$(pwd)/main_combined.csv"
  
  # Initialize the combined CSV file with header (will be overwritten later)
  : > "$COMBINED_CSV"
fi
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
LIGHT_RUNS=$(awk '/light_runs:/{flag=1;next}/^[^ ]/{flag=0}flag' $CONFIG_FILE | sed 's/ - //g')


PYTHON3=$(which python3)
hadd=$(which hadd)
# Loop over the main runs
IFS=$'\n' # Set Internal Field Separator to newline for iteration
for RUN in $MAIN_RUNS; do
  # Zero-pad the run number to form the directory name
  RUN=$(echo "$RUN" | tr -d ' ')
  PADDED_RUN=$RUN
  while [ ${#PADDED_RUN} -lt 4 ]; do
    PADDED_RUN="0$PADDED_RUN"
  done
  
  if [ $? -ne 0 ]; then
    echo "Failed to process run number: $RUN"
  fi

  # Construct the directory name
  DIR_NAME="${DIRECTORY_PATH}/main_run_${PADDED_RUN}"

  # Check if the directory exists
  if [ -d "$DIR_NAME" ]; then
    # Execute your commands here
    # For demonstration, I'm just printing the directory name
    echo "Executing commands in $DIR_NAME"
    if [ "$ANALYSIS_TYPE" == "merge-root" ]; then
      COMMAND=".$hadd $DIR_NAME/root/Run${RUN}_Point0.root $DIR_NAME/root/main_*.root"
      echo $COMMAND
      $hadd "$DIR_NAME/root/Run${RUN}_Point0.root" $DIR_NAME/root/main_*.root
      sleep 2 
    elif [ "$ANALYSIS_TYPE" == "combine-csv" ]; then
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
    elif [ "$ANALYSIS_TYPE" == "main-position" ]; then
      COMMAND="$PYTHON3 $(dirname $0)/new_coordinates.py $(dirname $0)/test.csv $DIR_NAME/root positions_main_run"
      $PYTHON3 $(dirname $0)/new_coordinates.py $(dirname $0)/test.csv $DIR_NAME/root positions_main_run
    fi
    # cd $DIR_NAME
    # Your commands here
  else
    echo "Directory $DIR_NAME not found, skipping..."
  fi
done

for RUN in $LIGHT_RUNS; do
  # Zero-pad the run number to form the directory name
  # PADDED_RUN=$(printf "%04d" $RUN 2>/dev/null)
  # PADDED_RUN=$(awk -v run="$RUN" 'BEGIN{printf "%04d", run}')
  RUN=$(echo "$RUN" | tr -d ' ')
  PADDED_RUN=$RUN
  while [ ${#PADDED_RUN} -lt 4 ]; do
    PADDED_RUN="0$PADDED_RUN"
  done
  
  if [ $? -ne 0 ]; then
    echo "Failed to process run number: $RUN"
  fi

  # Construct the directory name
  DIR_NAME="${DIRECTORY_PATH}/light_run_${PADDED_RUN}"

  # Check if the directory exists
  if [ -d "$DIR_NAME" ]; then
    # Execute your commands here
    # For demonstration, I'm just printing the directory name
    echo "Executing commands in $DIR_NAME"
    if [ "$ANALYSIS_TYPE" == "light-position" ]; then
      COMMAND="$PYTHON3 $(dirname $0)/new_coordinates.py $(dirname $0)/../datasets/lightRunPositions/light_run_${RUN}.csv $DIR_NAME/root positions_light_run"
      $PYTHON3 $(dirname $0)/new_coordinates.py $(dirname $0)/../datasets/lightRunPositions/light_run_${RUN}.csv $DIR_NAME/root positions_light_run
      echo $COMMAND
    fi
    # cd $DIR_NAME
    # Your commands here
  else
    echo "Directory $DIR_NAME not found, skipping..."
  fi
done

