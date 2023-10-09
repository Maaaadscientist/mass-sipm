#!/bin/bash

# Check the number of arguments passed to the script
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <config-file.yaml> <directory-path> <analysis-type>"
  exit 1
fi

# Initialize a global variable to keep track of the last update time
last_update_time=0
# Function to display progress bar
# Arguments: current_step, total_steps
show_progress_bar () {
  current_time=$(date +%s)
  if (( current_time - last_update_time >= 1 || last_update_time == 0 )); then
    last_update_time=$current_time
    current_step=$1
    total_steps=$2

    # Clear the line
    echo -ne "\r\033[K"

    # Calculate percentage
    percent=$(echo "scale=2; ($current_step / $total_steps) * 100" | bc)

    # Create progress bar
    num_bars=$(printf "%.0f" $(echo "scale=5; ($current_step / $total_steps) * 50" | bc)) # Increased to 50 bars
    #num_bars=$(echo "scale=1; 80 * ($current_step / $total_steps)" | bc)

    bar=""
    for ((i=0; i<$num_bars; i++)); do
      bar+=">"
    done
    for ((i=$num_bars; i<50; i++)); do  # Correspondingly increased to 50 bars
      bar+="-"
    done

    # Calculate elapsed time
    elapsed_time=$(($SECONDS - $start_time))
    if ((elapsed_time >= 3600)); then
      elapsed_str="$((elapsed_time/3600))h $((elapsed_time%3600/60))m $((elapsed_time%60))s"
    elif ((elapsed_time >= 60)); then
      elapsed_str="$((elapsed_time/60))m $((elapsed_time%60))s"
    else
      elapsed_str="$((elapsed_time))s"
    fi

    # Calculate estimated time
    estimated_time=$(printf "%.0f" $(echo "scale=10; ($elapsed_time / $current_step) * ($total_steps - $current_step)" | bc))
    if ((estimated_time >= 3600)); then
      estimated_str="$((estimated_time/3600))h $((estimated_time%3600/60))m $((estimated_time%60))s"
    elif ((estimated_time >= 60)); then
      estimated_str="$((estimated_time/60))m $((estimated_time%60))s"
    else
      estimated_str="$((estimated_time))s"
    fi

    if ((estimated_time != 0)); then
    # Print progress bar
      echo -ne "[$bar] ${percent}% | Elapsed Time: ${elapsed_str} | Estimated Remaining: ${estimated_str}\r"
    fi 
  fi
}

process_csv_file() {
  local input_file=$1
  local output_file=$2
  local header_captured=$3

  awk -v header="$header_captured" '
  BEGIN { FS=OFS="," }
  NR == 1 {
    for (i=1; i<=NF; i++) if ($i == "run") col=i;
    if (header == "false") print;
    next;
  }
  {
    if ($col ~ /^00[0-9A-Z]{2}$/) $col = substr($col,3,2);
    else if ($col ~ /^0[0-9A-Z]{3}$/) $col = substr($col,2,3);
    print;
  }
  ' "$input_file" >> "$output_file"
}

# Assign arguments to variables
CONFIG_FILE=$1
ANALYSIS_TYPE=$3

# List of allowed analysis types
# Declare an associative array for analysis type to suffix mapping
# Add new analysis type to the allowed types and its suffix
ALLOWED_ANALYSIS_TYPES=("merge-root" "combine-csv" "main-position" "light-position" "signal-fit" "vbd" "harvest" "produce-map" "final-harvest")

declare -A ANALYSIS_SUFFIXES
ANALYSIS_SUFFIXES=(["merge-root"]="main-reff" ["combine-csv"]="main-match" ["main-position"]="main-match" ["light-position"]="light-match" ["signal-fit"]="signal-fit" ["vbd"]="vbd" ["harvest"]="signal-fit" ["produce-map"]="produce-map" ["final-harvest"]="signal-fit")

# Check if the analysis type is allowed
if [[ ! " ${ALLOWED_ANALYSIS_TYPES[@]} " =~ " ${ANALYSIS_TYPE} " ]]; then
  echo "Error: Invalid analysis type '$ANALYSIS_TYPE'"
  echo "Allowed types are: ${ALLOWED_ANALYSIS_TYPES[@]}"
  exit 1
fi

# If the analysis type is "signal-fit", initialize the master CSV
if [[ "$ANALYSIS_TYPE" == "signal-fit" || "$ANALYSIS_TYPE" == "vbd" || $ANALYSIS_TYPE == "combine-csv" ]]; then
  MASTER_CSV="$(pwd)/all_merged_$ANALYSIS_TYPE.csv"
  : > "$MASTER_CSV"  # Initialize the master CSV file
  HEADER_CAPTURED=false
fi
# Fetch the suffix for the analysis type
SUFFIX=${ANALYSIS_SUFFIXES["$ANALYSIS_TYPE"]}

DIRECTORY_PATH="$(realpath $2)/$SUFFIX"
BASE_PATH="$(realpath $2)"
# Check if the given YAML file and directory exist
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Config file '$CONFIG_FILE' not found."
  exit 1
fi

#if [ ! -d "$DIRECTORY_PATH" ]; then
#  echo "Directory '$DIRECTORY_PATH' not found."
#  exit 1
#fi

# Extract the main_runs array from the YAML file using awk and sed
MAIN_RUNS=$(awk '/main_runs:/{flag=1;next}/^[^ ]/{flag=0}flag' $CONFIG_FILE | sed 's/ - //g')
LIGHT_RUNS=$(awk '/light_runs:/{flag=1;next}/^[^ ]/{flag=0}flag' $CONFIG_FILE | sed 's/ - //g')

total_lines=0
if [[ "$SUFFIX" == "main-match" || "$SUFFIX" == "signal-fit" || "$SUFFIX" == "vbd" || "$ANALYSIS_TYPE" == "produce-map" || "$ANALYSIS_TYPE" == "merge-root" ]]; then
  # Convert it to an array; assuming the elements are separated by newlines
  for RUN in $MAIN_RUNS; do
    ((total_lines++))
  done
fi
if [[ "$SUFFIX" == "light-match" || "$ANALYSIS_TYPE" == "produce-map" ]]; then
  for RUN in $LIGHT_RUNS; do
    ((total_lines++))
  done
fi

counter=0
# Record start time
start_time=$SECONDS
PYTHON3=$(which python3)
hadd=$(which hadd)
# Loop over the main runs
IFS=$'\n' # Set Internal Field Separator to newline for iteration
# Initialize the last update time to 0
last_update_time=0
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

  ((counter++))
  show_progress_bar $counter $total_lines
  # Construct the directory name
  DIR_NAME="${DIRECTORY_PATH}/main_run_${PADDED_RUN}"
  DIR_NAME_VBD="${BASE_PATH}/vbd/main_run_${PADDED_RUN}"
  DIR_NAME_DCR="${BASE_PATH}/new-dcr/main_run_${PADDED_RUN}"
  DIR_NAME_MATCH="${BASE_PATH}/main-match/main_run_${PADDED_RUN}"

  # Check if the directory exists
  if [[ -d "$DIR_NAME" || "$ANALYSIS_TYPE" == "produce-map" ]]; then
    # Execute your commands here
    # For demonstration, I'm just printing the directory name
    if [ "$ANALYSIS_TYPE" == "merge-root" ]; then
      if [ ! -d "$DIR_NAME/root" ]; then
        mkdir -p "$DIR_NAME/root"
      fi
      COMMAND=".$hadd $DIR_NAME/root/Run${RUN}_Point0.root $DIR_NAME/csv/main_*.root"
      $hadd "$DIR_NAME/root/Run${RUN}_Point0.root" $DIR_NAME/csv/main_*.root &> /dev/null
      sleep 2
      rm -rf $DIR_NAME/csv/main_*.root
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
      fi
    elif [ "$ANALYSIS_TYPE" == "main-position" ]; then
      COMMAND="$PYTHON3 $(dirname $0)/new_coordinates.py $(dirname $0)/test.csv $DIR_NAME/root positions_main_run"
      $PYTHON3 $(dirname $0)/new_coordinates.py $(dirname $0)/test.csv $DIR_NAME/root positions_main_run
    # Handle the new analysis type
    elif [ "$ANALYSIS_TYPE" == "signal-fit" ]; then
      # Construct the CSV directory path
      CSV_DIR="$DIR_NAME/csv"

      # Check if directory exists
      if [ -d "$CSV_DIR" ]; then
        # Check if there are 96 CSV files
        NUM_CSV_FILES=$(ls $CSV_DIR/*.csv 2>/dev/null | wc -l)
        if [ "$NUM_CSV_FILES" -ne 96 ]; then
          echo "Expected 96 CSV files in $CSV_DIR but found $NUM_CSV_FILES. for run $RUN Skipping..."
          #continue
        fi

        # Merge the CSV files into the master CSV
        for file in $CSV_DIR/*.csv; do
          process_csv_file "$file" "$MASTER_CSV" "$HEADER_CAPTURED"
          if ! $HEADER_CAPTURED; then
          #  tail -n +2 $file >> $MASTER_CSV
          #else
          #  cat $file > $MASTER_CSV
            HEADER_CAPTURED=true
          fi
        done
      else
        echo "CSV directory $CSV_DIR not found, skipping..."
      fi
    #fi
    elif [ "$ANALYSIS_TYPE" == "vbd" ]; then
      # Construct the CSV directory path
      CSV_DIR="$DIR_NAME/csv"

      # Check if directory exists
      if [ -d "$CSV_DIR" ]; then
        # Check if there are 96 CSV files
        NUM_CSV_FILES=$(ls $CSV_DIR/*.csv 2>/dev/null | wc -l)
        if [ "$NUM_CSV_FILES" -ne 1 ]; then
          echo "Expected 1 CSV file in $CSV_DIR but found $NUM_CSV_FILES. for run $RUN Skipping..."
          #continue
        fi

        # Merge the CSV files into the master CSV
        for file in $CSV_DIR/*.csv; do
          process_csv_file "$file" "$MASTER_CSV" "$HEADER_CAPTURED"
          if ! $HEADER_CAPTURED; then
          #  tail -n +2 $file >> $MASTER_CSV
          #else
          #  cat $file > $MASTER_CSV
            HEADER_CAPTURED=true
          fi
        done
      else
        echo "CSV directory $CSV_DIR not found, skipping..."
      fi
    #fi
    elif [ "$ANALYSIS_TYPE" == "harvest" ]; then
      # Construct the CSV directory path
      CSV_DIR="$DIR_NAME/csv"
      CSV_DIR2="$DIR_NAME_VBD/csv"
      # Check if directory exists
      if [ -d "$CSV_DIR" ]; then
        # Check if there are 96 CSV files
        NUM_CSV_FILES=$(ls $CSV_DIR/*.csv 2>/dev/null | wc -l)
        if [ "$NUM_CSV_FILES" -ne 96 ]; then
          echo "Expected 96 CSV files in $CSV_DIR but found $NUM_CSV_FILES. for run $RUN "
          #continue
        fi
        $PYTHON3 $(dirname $0)/combine_two_csv.py $CSV_DIR $CSV_DIR2 signal-fit-harvests/csv/run_${PADDED_RUN}.csv   
        $PYTHON3 $(dirname $0)/csv2dataframe.py signal-fit-harvests/csv/run_${PADDED_RUN}.csv signal-fit-harvests/root/run_${PADDED_RUN}.root
      else
        echo "CSV directory $CSV_DIR not found, skipping..."
      fi
    elif [ "$ANALYSIS_TYPE" == "final-harvest" ]; then
      # Construct the CSV directory path
      CSV_DIR_SIGNAL="$DIR_NAME/csv"
      CSV_DIR_VBD="$DIR_NAME_VBD/csv"
      CSV_DIR_DCR="$DIR_NAME_DCR/csv"
      CSV_DIR_MATCH="$DIR_NAME_MATCH/root"
      # Check if directory exists
      if [ -d "$CSV_DIR_SIGNAL" ]; then
        # Check if there are 96 CSV files
        NUM_CSV_FILES=$(ls $CSV_DIR_SIGNAL/*.csv 2>/dev/null | wc -l)
        if [ "$NUM_CSV_FILES" -ne 96 ]; then
          echo "Expected 96 CSV files in $CSV_DIR_SIGNAL but found $NUM_CSV_FILES. for run $RUN "
          #continue
        fi
        $PYTHON3 $(dirname $0)/combine_csv_files.py $CSV_DIR_SIGNAL $CSV_DIR_VBD final-harvests/tmp/tmp_1_${PADDED_RUN}.csv
        $PYTHON3 $(dirname $0)/combine_csv_files.py final-harvests/tmp/tmp_1_${PADDED_RUN}.csv $CSV_DIR_DCR final-harvests/tmp/tmp_2_${PADDED_RUN}.csv
        $PYTHON3 $(dirname $0)/combine_csv_files.py final-harvests/tmp/tmp_2_${PADDED_RUN}.csv $CSV_DIR_MATCH final-harvests/tmp/tmp_3_${PADDED_RUN}.csv -k run -d chi2,var,point -n x,match_x -n y,match_y -n x_left_error,match_x_err_down -n x_right_error,match_x_err_up -n y_up_error,match_y_err_up -n y_down_error,match_y_err_down
        $PYTHON3 $(dirname $0)/get_tile_sn.py -r $RUN -o final-harvests/sn/sn_${PADDED_RUN}.csv
        $PYTHON3 $(dirname $0)/combine_csv_files.py final-harvests/tmp/tmp_3_${PADDED_RUN}.csv final-harvests/sn/sn_${PADDED_RUN}.csv final-harvests/csv/run_${PADDED_RUN}.csv -k run,pos
        $PYTHON3 $(dirname $0)/csv2dataframe.py final-harvests/csv/run_${PADDED_RUN}.csv final-harvests/root/run_${PADDED_RUN}.root
      else
        echo "CSV directory $CSV_DIR not found, skipping..."
      fi
    elif [ "$ANALYSIS_TYPE" == "produce-map" ]; then
      COMMAND="$PYTHON3 $(dirname $0)/produce_light_map.py $(dirname $0)/../config/produce_light_map.yaml $(dirname $0)/map_outputs main ${RUN}"
      $PYTHON3 $(dirname $0)/produce_light_map.py $(dirname $0)/../config/produce_light_map.yaml $(dirname $0)/map_outputs main ${RUN}
    fi
    # cd $DIR_NAME
    # Your commands here
  else
    echo "Directory $DIR_NAME not found, skipping..."
  fi
  
done

# Initialize the last update time to 0
#last_update_time=0
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

  ((counter++))
  show_progress_bar $counter $total_lines
  # Construct the directory name
  DIR_NAME="${DIRECTORY_PATH}/light_run_${PADDED_RUN}"

  # Execute your commands here
  # For demonstration, I'm just printing the directory name
  if [ "$ANALYSIS_TYPE" == "light-position" ]; then
    COMMAND="$PYTHON3 $(dirname $0)/new_coordinates.py $(dirname $0)/../datasets/lightRunPositions/light_run_${RUN}.csv $DIR_NAME/root positions_light_run"
    $PYTHON3 $(dirname $0)/new_coordinates.py $(dirname $0)/../datasets/lightRunPositions/light_run_${RUN}.csv $DIR_NAME/csv positions_light_run
  elif [ "$ANALYSIS_TYPE" == "produce-map" ]; then
    COMMAND="$PYTHON3 $(dirname $0)/produce_light_map.py $(dirname $0)/../config/produce_light_map.yaml $(dirname $0)/map_outputs light ${RUN}"
    $PYTHON3 $(dirname $0)/produce_light_map.py $(dirname $0)/../config/produce_light_map.yaml $(dirname $0)/map_outputs light ${RUN}
  fi
  # cd $DIR_NAME
  # Your commands here
done

