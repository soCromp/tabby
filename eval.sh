#!/bin/bash

# Check if directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

# Iterate over all 'samplesclean.csv' files in subdirectories
find "$1" -type f -name 'samplesclean.csv' | while read -r filepath; do
  # Get the enclosing directory of the file
  dirpath=$(dirname "$filepath")
  
  # Print the full path of the enclosing directory
  echo "Processing directory: $dirpath"
  
  # Call the Python script ~/eval.py with the file as an argument
  python3 ~/be_great/eval.py adult "$dirpath"
done

