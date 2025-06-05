import json
import glob
import os
from tqdm import tqdm
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description='Filter dataset based on depth.')

# Add an argument for the depth value
parser.add_argument('--depth', type=int, default=0, help='Depth value for filtering')
# Add an argument for using only attributes
parser.add_argument('--only_attributes', action='store_true', help='Use only attributes')
# Add an argument for excluding negatives
parser.add_argument('--no_negatives', action='store_true', help='Exclude negatives')

# Parse the command line arguments
args = parser.parse_args()

# Assign the depth value to the DEPTH variable
DEPTH = args.depth
ONLY_ATTRIBUTES = args.only_attributes
NO_NEGATIVES = args.no_negatives

# Define the prefix based on the arguments
prefix = "Att" if ONLY_ATTRIBUTES else "Rel"
prefix += "Noneg" if NO_NEGATIVES else "Neg"


# Define the path to the folder containing the JSONL files
folder_path = r"path/to/depth-n/" # e.g. rf"...\rule-reasoning-dataset-V2020.2.5\rule-reasoning-dataset-V2020.2.5.0\original\depth-{DEPTH}"

# Get the list of JSONL files in the folder
jsonl_files = glob.glob(folder_path + "/*.jsonl")

# Create a new folder named "original" in the folder_path
original_folder_path = os.path.join(folder_path, "original")
os.makedirs(original_folder_path, exist_ok=True)

# Move all jsonl_files to the "original" folder
for file_path in jsonl_files:
    new_file_path = os.path.join(original_folder_path, os.path.basename(file_path))
    os.rename(file_path, new_file_path)

# Update the jsonl_files list with the new file paths
jsonl_files = glob.glob(original_folder_path + "/*.jsonl")

# Create a new folder named "filtered" in the parent folder of the original files
filtered_folder_path = os.path.join(folder_path, f"filtered_{prefix}")
os.makedirs(filtered_folder_path, exist_ok=True)


# Iterate over each JSONL file
for file_path in tqdm(jsonl_files, desc="Files"):
    # Read the contents of the JSONL file
    filtered_entries = []
    with open(file_path, "r") as file:
        for line in tqdm(file, desc="Examples", leave=False):
            # Parse the JSON line
            entry = json.loads(line)

            # Filter the entries based on the "id" value
            if entry.get("id", "").startswith(prefix):
                filtered_entries.append(entry)

    # Define the path for the new JSONL file
    new_file_path = os.path.join(filtered_folder_path, os.path.basename(file_path).replace(".jsonl", "_filtered.jsonl"))

    # Write the filtered entries to the new JSONL file
    with open(new_file_path, "w") as new_file:
        for entry in tqdm(filtered_entries, desc="Writing", leave=False):
            new_file.write(json.dumps(entry) + "\n")
