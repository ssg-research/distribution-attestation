#!/bin/bash

# Authors: Vasisht Duddu, Anudeep Das, Nora Khayata, Hossein Yalame, Thomas Schneider, N Asokan
# Copyright 2024 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Specify the directory path
directory_path="."

# Loop through files
for file in "$directory_path"/*CENSUS*_s_*verify.log; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        
        # Print the first line of the file
        first_line=$(head -n 1 "$file")
        echo "First line: $first_line"
        
        # Print line that starts with "Normalized number of rejects at"
        normalized_line=$(grep -m 1 "^Normalized number of false accepts ar frr5:" "$file")
        echo "Normalized line: $normalized_line"
        
        echo "-------------------"
    fi
done
