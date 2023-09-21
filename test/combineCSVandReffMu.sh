#!/bin/bash

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
# Input and output CSV files
input_csv=$1
output_csv="output.csv"

# Get total number of lines in the input CSV (excluding the header)
total_lines=$(($(wc -l < "$input_csv") - 1))

# Initialize the output CSV with headers
echo "run_type,run_id,point,new_x,new_y,total_time,mu_index,mu_value,mu_error" > "$output_csv"
# Initialize counter
counter=0

# Record start time
start_time=$SECONDS

# Initialize the last update time to 0
last_update_time=0
# Skip the first line (headers) and read each line from the input CSV
tail -n +2 "$input_csv" | while read -r line; do
  # Increment counter
  ((counter++))

  # Display progress bar
  show_progress_bar $counter $total_lines
  # Progress bar
  #echo -ne "Processing: $counter/$total_lines\r"
  # Extract columns from the line
  IFS=',' read -ra cols <<< "$line"
  run_type="${cols[0]}"
  run_id="${cols[1]}"
  point="${cols[2]}"
  new_x="${cols[3]}"
  new_y="${cols[4]}"
  total_time="${cols[5]}"

  PADDED_RUN=$run_id
  while [ ${#PADDED_RUN} -lt 4 ]; do
    PADDED_RUN="0$PADDED_RUN"
  done
  # Decide the root file directory based on run_type
  if [ "$run_type" == "light" ]; then
    root_file_dir="/junofs/users/wanghanwen/match_light_field/light-match/light_run_${PADDED_RUN}/root/"
  elif [ "$run_type" == "main" ]; then
    root_file_dir="/junofs/users/wanghanwen/match_light_field/main-match/main_run_${PADDED_RUN}/root/"
  else
    echo "Unknown run_type: $run_type. Skipping..."
    continue
  fi

  # Create ROOT file path
  root_file_path="${root_file_dir}Run${run_id}_Point${point}.txt"

  # Run the getReffMu binary to get the mu values and errors
  while read -r mu_line; do
    IFS=' ' read -ra mu_cols <<< "$mu_line"
    mu_index="${mu_cols[0]}"
    mu_value="${mu_cols[1]}"
    mu_error="${mu_cols[2]}"

    # Append the row to the output CSV
    echo "$run_type,$run_id,$point,$new_x,$new_y,$total_time,$mu_index,$mu_value,$mu_error" >> "$output_csv"

  done < <(cat "$root_file_path")

done

# Done
echo -e "\nProcessing complete."
