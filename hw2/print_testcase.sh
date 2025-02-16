#!/bin/bash

# Usage: ./script.sh /path/to/directory

# Check if directory path is provided
directory=${1:-.}  # Use current directory if no path provided

for file in "$directory"/*.txt; do
    if [ -f "$file" ]; then
        echo -n "$(basename "$file"): "
        awk '{print $6; exit}' "$file"
    fi
done