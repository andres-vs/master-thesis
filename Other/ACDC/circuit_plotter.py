import json
import re
import networkx as nx
import pygraphviz as pgv
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
print(len(filtered_pairs))
# Initialize a directed graph
G = nx.DiGraph()

# Add edges to the graph
for start, end in filtered_pairs:
    G.add_edge(start, end)

# Create a dictionary to store node labels and colors
labels = {}
node_colors = {}

# Function to extract layer and head/node number
def extract_layer_and_number(node):
    layer_match = re.search(r'blocks\.(\d+)\.', node)
    layer = int(layer_match.group(1)) if layer_match else 0
    if 'attn' in node:
        number_match = re.search(r'attn\.hook_(result|q|k|v|q_input|k_input|v_input)\[:, :, (\d+)\]', node)
        number = int(number_match.group(2)) if number_match else 0
    elif 'hook_v_input' in node or 'hook_k_input' in node or 'hook_q_input' in node:
        number_match = re.search(r'hook_(v|k|q)_input\[:, :, (\d+)\]', node)
        number = int(number_match.group(2)) if number_match else 0
    else:
        number = 0  # MLP nodes have only one node per layer
    return layer, number

# Determine labels and colors for each node
for node in G.nodes():
    layer, number = extract_layer_and_number(node)
    if 'hook_resid_pre' in node:
        labels[node] = 'embedding_output'
        node_colors[node] = 'lightcoral'
    elif 'hook_resid_post' in node:
        labels[node] = 'encoder_output'
        node_colors[node] = 'lightcoral'
    elif 'attn' in node:
        labels[node] = f'a{layer}.{number}'
        node_colors[node] = 'skyblue'
    elif 'mlp' in node:
        labels[node] = f'm{layer}'
        node_colors[node] = 'lightgreen'
    else:
        labels[node] = node
        node_colors[node] = 'lightgrey'

# Create a PyGraphviz AGraph
A = pgv.AGraph(directed=True)

# Add nodes and edges to the AGraph
for node in G.nodes():
    A.add_node(node, label=labels[node], style='filled', fillcolor=node_colors[node], shape='rect', width=0.5, height=0.5)
for start, end in filtered_pairs:
    A.add_edge(start, end)

# Draw the graph to a file
A.layout(prog='dot')
A.draw('circuit_plot.png')

# Display the graph using matplotlib
img = plt.imread('circuit_plot.png')
plt.figure(figsize=(12, 26))
plt.imshow(img)
plt.axis('off')
plt.title('Circuit Plot')
plt.show()