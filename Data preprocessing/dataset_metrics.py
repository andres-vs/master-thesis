import glob
import json
from tqdm import tqdm
from transformers import AutoTokenizer

DEPTH = 1
prefix = "prop_noneg"

# Define the path to the folder containing the JSONL files
folder_path = r"path/to/depth-n/filtered/formated" # e.g. rf"...\rule-reasoning-dataset-V2020.2.5\rule-reasoning-dataset-V2020.2.5.0\original\depth-{DEPTH}"\filtered_{prefix}\formatted"
print(folder_path)
# Get the list of JSONL files in the folder
jsonl_files = glob.glob(folder_path + "/*.jsonl")




def get_average_character_length(jsonl_files):
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
            print(f"Average length of input entries in {file_path}: {average_length}")
        else:
            print(f"No entries found in {file_path}.")

def get_average_token_length(jsonl_files, tokenizer):
    for file_path in tqdm(jsonl_files, desc="Files"):
        total_tokens = 0
        num_entries = 0
        with open(file_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                input_tokens = tokenizer(entry['input'])['input_ids']
                total_tokens += len(input_tokens)
                num_entries += 1

        if num_entries > 0:
            average_tokens = total_tokens / num_entries
            print(f"Average token length of input entries in {file_path}: {average_tokens}")
        else:
            print(f"No entries found in {file_path}.")

if __name__ == "__main__":
    # Load the "bert-base-uncased" tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    get_average_token_length(jsonl_files, tokenizer)