import glob
import json
from tqdm import tqdm

DEPTH = 0
prefix = "prop_noneg"

# Define the path to the folder containing the JSONL files
folder_path = rf"C:\Users\andre\Documents\School\Hoger\Masterproef\Data\rule-reasoning-dataset-V2020.2.5\rule-reasoning-dataset-V2020.2.5.0\original\depth-{DEPTH}\filtered_{prefix}\formatted"
print(folder_path)
# Get the list of JSONL files in the folder
jsonl_files = glob.glob(folder_path + "/*.jsonl")




for file_path in tqdm(jsonl_files, desc="Files"):
    total_length = 0
    num_entries = 0
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            input_length = len(entry['input'])
            total_length += input_length
            num_entries += 1

    if num_entries > 0:
        average_length = total_length / num_entries
        print(f"Average length of input entries: {average_length}")
    else:
        print("No entries found.")