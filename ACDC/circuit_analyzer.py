import json
import re
import os

THRESHOLD = 0.01585
# Define the input file path
input_file_path = 'post_processing/acdc_results_greater-than_GPT2_t0.01585.json'

# Extract the variable part of the input file name
variable_part = re.search(r'acdc_results_(.*)\.json', input_file_path).group(1)

# Load the JSON data
with open(input_file_path, 'r') as file:
    data = json.load(file)['data']

text = data[0]['text']
y_values = data[0]['y']

# Initialize a list to store the filtered node pairs
filtered_pairs = []

# Iterate through the data to filter the node pairs
for edge, y_value in zip(text, y_values):
    if y_value >= THRESHOLD:
        [start, end] = edge.split(' to ')
        filtered_pairs.append((start, end))

print(f"Number of edges: {len(filtered_pairs)}")

# Extract all unique nodes from the filtered pairs
unique_nodes = set()
for start, end in filtered_pairs:
    unique_nodes.add(start)
    unique_nodes.add(end)

print(f"Number of unique nodes: {len(unique_nodes)}")

# Rename nodes according to the specified rules
node_mapping = {}
for node in unique_nodes:
    match_attn = re.match(r'blocks\.(\d+)\..*\[:, :, (\d+)\]', node)
    match_mlp = re.match(r'blocks\.(\d+)\.hook_mlp.*', node)
    match_resid_pre = re.match(r'.*hook_resid_pre.*', node)
    match_resid_post = re.match(r'.*hook_resid_post.*', node)
    
    if match_attn:
        layer = match_attn.group(1)
        head = match_attn.group(2)
        new_name = f'a{layer}h{head}'
    elif match_mlp:
        layer = match_mlp.group(1)
        new_name = f'm{layer}'
    elif match_resid_pre:
        new_name = 'embedding'
    elif match_resid_post:
        new_name = 'output'
    else:
        new_name = node
    node_mapping[node] = new_name

# Extract all unique renamed nodes
unique_renamed_nodes = set(node_mapping.values())

# Apply the node renaming to the filtered pairs and filter out self-loops
unique_renamed_pairs = [(node_mapping[start], node_mapping[end]) for start, end in filtered_pairs if node_mapping[start] != node_mapping[end]]

# Sort the unique nodes based on their layer number
def extract_layer_number(node):
    match = re.search(r'\d+', node)
    return int(match.group()) if match else float('inf')

sorted_unique_nodes = sorted(unique_renamed_nodes, key=extract_layer_number)

print(f"Number of unique renamed nodes: {len(unique_renamed_nodes)}")
print(f"Number of renamed pairs: {len(unique_renamed_pairs)}")

print("Sorted unique nodes based on their layer number:")
print(sorted_unique_nodes)

print("Renamed pairs:")
print(unique_renamed_pairs)

# Define the output file path
output_file_path = f'post_processing/results_{variable_part}_renamed.json'

# Save the renamed pairs to the output file
with open(output_file_path, 'w') as file:
    json.dump({'nodes': sorted_unique_nodes, 'edges': unique_renamed_pairs}, file)

print(f"Renamed pairs saved to {output_file_path}")