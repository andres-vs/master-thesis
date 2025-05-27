import glob
import json
import os
from tqdm import tqdm
import argparse

def generate_hf_dataset_from_jsonl(jsonl_file, output_file, filter_qdep=None, depth=0, filter_rconc=False, filter_ps=None):
    examples = []

    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            context = data["context"]
            questions = data['questions']
            for question in questions:
                if filter_qdep and question['meta']['QDep'] != filter_qdep:
                    continue
                if filter_rconc and "rconc" in question['meta']['strategy']:
                    continue
                if filter_ps and question['meta']['strategy'] != filter_ps:
                    continue
                else:
                    example = {
                        'input': context + "[SEP]" + question['text'],
                        'label': question['label'],
                        'depth': question['meta']['QDep'],
                        'proof_strategy': question['meta']['strategy'],
                    }
                    examples.append(example)

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')


# Create an argument parser
parser = argparse.ArgumentParser(description='Filter dataset based on depth.')

# Add an argument for the depth value
parser.add_argument('--depth', type=int, default=0, help='Depth value for filtering')
# Add an argument for using only attributes
parser.add_argument('--only_attributes', action='store_true', help='Use only attributes')
# Add an argument for excluding negatives
parser.add_argument('--no_negatives', action='store_true', help='Exclude negatives')
# Add an argument for filtering on QDep
parser.add_argument('--RD', type=int, help='Filter on reasoning depth')
# Add an argument for filtering on no rconc
parser.add_argument('--no_rconc', action='store_true', help='Filter on rconc or inv-rconc (no rconc)')
# Add an argument for filtering on proof strategies
parser.add_argument('--PS', type=str, help='Filter on proof strategies (only proof, inv-proof, random or inv-random, rconc or inv-rconc)')

# Parse the command line arguments
args = parser.parse_args()

# Assign the depth value to the DEPTH variable
DEPTH = args.depth
ONLY_ATTRIBUTES = args.only_attributes
NO_NEGATIVES = args.no_negatives
FILTER_QDEP = args.RD
FILTER_RCONC = args.no_rconc
FILTER_PS = args.PS

# Define the prefix based on the arguments
prefix = "Att" if ONLY_ATTRIBUTES else "Rel"
prefix += "Noneg" if NO_NEGATIVES else "Neg"


# Define the path to the folder containing the JSONL files
folder_path = rf"C:\Users\andre\Documents\School\Hoger\Masterproef\Data\rule-reasoning-dataset-V2020.2.5\rule-reasoning-dataset-V2020.2.5.0\original\depth-{DEPTH}\filtered_{prefix}"
print(folder_path)
# Get the list of JSONL files in the folder
jsonl_files = glob.glob(folder_path + "/*.jsonl")

dir_name = f"formatted"
if FILTER_QDEP:
    dir_name += f"_RD-{DEPTH}"
if FILTER_RCONC:
    dir_name += "_no_rconc"
if FILTER_PS:
    dir_name += f"_PS-{FILTER_PS}"

formatted_folder_path = os.path.join(folder_path, dir_name)


os.makedirs(formatted_folder_path, exist_ok=True)

# Iterate over each JSONL file that does not have "meta" in its name
for file_path in tqdm(jsonl_files, desc="Files"):
    if "meta" not in file_path:
        # Define the path for the new JSONL file
        new_file_path = os.path.join(formatted_folder_path, os.path.basename(file_path).replace(".jsonl", "_formatted.jsonl"))    
        generate_hf_dataset_from_jsonl(file_path, new_file_path, filter_qdep=FILTER_QDEP, depth=DEPTH, filter_rconc=FILTER_RCONC, filter_ps=FILTER_PS)
