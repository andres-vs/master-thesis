import json
import re
import networkx as nx
import matplotlib.pyplot as plt

# Load the JSON data
with open('results_acdc_text-entailment_BERT_0.02.json', 'r') as file:
    data = json.load(file)['data']

text = data[0]['text']
y_values = data[0]['y']

# Initialize a list to store the filtered node pairs
filtered_pairs = []

# Iterate through the data to filter the node pairs
for edge, y_value in zip(text, y_values):
    if y_value >= 0.02:
        [start, end] = edge.split(' to ')
        filtered_pairs.append((start, end))

# # Store the filtered pairs for later use
# with open('filtered_pairs.json', 'w') as file:
#     json.dump(filtered_pairs, file)

# print("Filtered pairs have been stored.")

# Load the filtered pairs
with open('filtered_pairs.json', 'r') as file:
    filtered_pairs = json.load(file)

# Initialize a directed graph
G = nx.DiGraph()

# Add edges to the graph
for start, end in filtered_pairs:
    G.add_edge(start, end)

# Create a dictionary to store node positions and labels
pos = {}
labels = {}
node_colors = []

# Function to extract layer and head/node number
def extract_layer_and_number(node):
    layer = int(re.search(r'blocks\.(\d+)\.', node).group(1))
    if 'attn' in node:
        number = int(re.search(r'attn\.hook_(result|q|k|v|q_input|k_input|v_input)\[:, :, (\d+)\]', node).group(2))
    elif 'hook_v_input' in node or 'hook_k_input' in node or 'hook_q_input' in node:
        number = int(re.search(r'hook_(v|k|q)_input\[:, :, (\d+)\]', node).group(2))
        # print('Node:', node)
        # print('Number:', number)
        # print('number type:', type(number))
    else:
        number = 0  # MLP nodes have only one node per layer
    return layer, number

# Determine positions and labels for each node
for node in G.nodes():
    if 'hook_resid_pre' in node:
        pos[node] = (6, -26)  # Embedding layer
        labels[node] = 'embedding_output'
        node_colors.append('lightcoral')
    elif 'hook_resid_post' in node:
        pos[node] = (6, 0)  # Output layer
        labels[node] = 'encoder_output'
        node_colors.append('lightcoral')
    else:
        # print('Node:', node)
        layer, number = extract_layer_and_number(node)
        if 'attn' in node or 'hook_v_input' in node or 'hook_k_input' in node or 'hook_q_input' in node:
            # print('Node:', node)
            # print('Layer type:', type(layer))
            # print('Number type:', type(number))
            pos[node] = (number, -layer * 2)  # Position: (head number, -layer * 2) for top to bottom
            labels[node] = f'a{layer}.{number}'
            node_colors.append('skyblue')
        elif 'mlp' in node:	
            pos[node] = (6, -layer * 2 - 1)  # Center MLP nodes
            labels[node] = f'm{layer}'
            node_colors.append('lightgreen')
        
# Draw the graph
plt.figure(figsize=(12, 26))
nx.draw(G, pos, labels=labels, with_labels=True, node_size=1000, node_color=node_colors, font_size=10, font_weight='bold', arrowsize=20, node_shape='s')
plt.title('Circuit Plot')
plt.show()