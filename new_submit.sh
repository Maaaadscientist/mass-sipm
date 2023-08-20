#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_path>"
    exit 1
fi

input_path="$1"

if [ ! -e "$input_path" ]; then
    echo "Error: Input path does not exist."
    exit 1
fi

find_root_files() {
    local start_path="$1"
    local root_files=()

    if [ -d "$start_path" ]; then
        while IFS= read -r -d '' file; do
            if [ "${file##*.}" = "root" ]; then
                root_files+=("$file")
            fi
        done < <(find "$start_path" -type f -name "*.root" -print0)
    elif [ -f "$start_path" ] && [ "${start_path##*.}" = "root" ]; then
        root_files+=("$start_path")
    fi

    printf '%s\n' "${root_files[@]}"
}

for root_file in $(find_root_files "$input_path"); do
    root_file_parent_dir=$(dirname "$root_file")
    hep_sub doGeoFit.sh -e /dev/null -o /dev/null -argu "$root_file_parent_dir"
done

