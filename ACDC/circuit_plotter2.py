import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

# Load the JSON data
with open('post_processing/results_greater-than_GPT2_t0.01585_renamed.json', 'r') as file:
    data = json.load(file)

nodes = data.get('nodes', [])
edges = data.get('edges', [])

print(f"Total nodes: {len(nodes)}")
print(f"Total edges: {len(edges)}")

# Function to extract layer number from node name
def extract_layer_number(node):
    match = re.search(r'\d+', node)
    return int(match.group()) if match else float('inf')

# Initialize the graph
G = nx.DiGraph()

# Add special nodes
G.add_node("embedding", type='embedding')
G.add_node("output", type='output')

# Remove 'embedding' and 'output' from nodes list if present
nodes_without_special = [node for node in nodes if node not in ['embedding', 'output']]

# Group nodes by layer number
layer_dict = {}
for node in nodes_without_special:
    layer = extract_layer_number(node)
    if layer not in layer_dict:
        layer_dict[layer] = []
    layer_dict[layer].append(node)

# Sort layers in ascending order
sorted_layers = sorted(layer_dict.keys())
print(f"Sorted layers: {sorted_layers}")

# Add nodes to the graph, sorted within layers
for layer in sorted_layers:
    layer_nodes = layer_dict[layer]
    # Sort nodes: 'a' nodes first, then 'm' nodes; within each group, sort alphabetically
    layer_nodes_sorted = sorted(layer_nodes, key=lambda x: (0 if x.startswith('a') else 1, x))
    for node in layer_nodes_sorted:
        node_type = 'attention' if node.startswith('a') else 'mlp'
        G.add_node(node, type=node_type, layer=layer)

# Add edges from the JSON data
for start, end in edges:
    # Exclude 'embedding' and 'output' from this step
    if start not in ['embedding', 'output'] and end not in ['embedding', 'output']:
        if G.has_node(start) and G.has_node(end):
            G.add_edge(start, end)
        else:
            print(f"Warning: Edge ({start}, {end}) includes unknown nodes.")

# Connect 'embedding' to the first layer nodes
if sorted_layers:
    first_layer_nodes = layer_dict.get(sorted_layers[0], [])
    for node in first_layer_nodes:
        G.add_edge("embedding", node)

# Connect the last layer nodes to 'output'
if sorted_layers:
    last_layer_nodes = layer_dict.get(sorted_layers[-1], [])
    for node in last_layer_nodes:
        G.add_edge(node, "output")

# Assign positions using Graphviz layout
try:
    pos = graphviz_layout(G, prog='dot')
except:
    print("Graphviz layout failed. Ensure Graphviz is installed and added to PATH.")
    pos = nx.spring_layout(G)

# Define node colors based on type
color_map = []
for node in G:
    if G.nodes[node].get('type') == 'embedding':
        color_map.append('gold')
    elif G.nodes[node].get('type') == 'output':
        color_map.append('red')
    elif G.nodes[node].get('type') == 'attention':
        color_map.append('skyblue')
    elif G.nodes[node].get('type') == 'mlp':
        color_map.append('lightgreen')
    else:
        color_map.append('grey')  # Default color

# Define node shapes based on type
node_shapes = {}
for node in G:
    if G.nodes[node].get('type') == 'embedding':
        node_shapes[node] = 'diamond'
    elif G.nodes[node].get('type') == 'output':
        node_shapes[node] = 'doublecircle'
    elif G.nodes[node].get('type') == 'attention':
        node_shapes[node] = 'ellipse'
    elif G.nodes[node].get('type') == 'mlp':
        node_shapes[node] = 'box'
    else:
        node_shapes[node] = 'circle'  # Default shape

# Start plotting
plt.figure(figsize=(15, 10))
plt.title("BERT Subgraph Visualization: Attention Heads and MLP Layers", fontsize=16)

# Draw edges first
nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20, edge_color='black')

# Draw nodes with different shapes
unique_shapes = set(node_shapes.values())

for shape in unique_shapes:
    shaped_nodes = [node for node in G if node_shapes[node] == shape]
    if not shaped_nodes:
        continue  # Skip if no nodes of this shape

    # Determine node_shape parameter for matplotlib
    if shape == 'ellipse':
        nx_shape = 'o'  # Circle
    elif shape == 'box':
        nx_shape = 's'  # Square
    elif shape == 'diamond':
        nx_shape = 'D'  # Diamond
    elif shape == 'doublecircle':
        # Matplotlib does not support doublecircle directly; using 'o' with edge colors
        nx_shape = 'o'
    else:
        nx_shape = 'o'  # Default to circle

    # For 'doublecircle', add edge colors or sizes to differentiate
    if shape == 'doublecircle':
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=shaped_nodes,
            node_color=[color_map[list(G).index(node)] for node in shaped_nodes],
            node_shape=nx_shape,
            node_size=1200,
            edgecolors='black',
            linewidths=2,
            alpha=0.9
        )
    else:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=shaped_nodes,
            node_color=[color_map[list(G).index(node)] for node in shaped_nodes],
            node_shape=nx_shape,
            node_size=800,
            alpha=0.9
        )

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

# Create a legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='D', color='w', label='Embedding', markerfacecolor='gold', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Output', markerfacecolor='red', markeredgecolor='black', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Attention Head (a nodes)', markerfacecolor='skyblue', markersize=10),
    Line2D([0], [0], marker='s', color='w', label='MLP Layer (m nodes)', markerfacecolor='lightgreen', markersize=10)
]

plt.legend(handles=legend_elements, loc='upper right')

plt.axis('off')
plt.tight_layout()

# Save the graph to a file
plt.savefig("bert_subgraph.png", format="PNG")
plt.show()
