import plotly.graph_objects as go
import numpy as np
import json
import re
import pygraphviz as pgv
import matplotlib.pyplot as plt

# Sample classification head weights for each class
# Replace these with the actual weights for entailment and non-entailment
weights_entailment = np.random.randn(768)  # Example data, replace with actual weightsquit()
weights_non_entailment = np.random.randn(768)  # Example data, replace with actual weights

# Create x-axis labels (indices of the components)
indices = list(range(len(weights_entailment)))

# Create the bar chart for entailment weights
fig = go.Figure()

fig.add_trace(go.Bar(
    x=indices,
    y=weights_entailment,
    name="Entailment Class",
    hoverinfo="x+y",
    text=[f"Index: {i}" for i in indices],  # Text that shows up when hovering
    hovertext=[f"Component {i}: {weight:.2f}" for i, weight in enumerate(weights_entailment)],
    marker=dict(color='blue')
))

# Create the bar chart for non-entailment weights below the entailment chart
fig.add_trace(go.Bar(
    x=indices,
    y=weights_non_entailment,
    name="Non-Entailment Class",
    hoverinfo="x+y",
    text=[f"Index: {i}" for i in indices],
    hovertext=[f"Component {i}: {weight:.2f}" for i, weight in enumerate(weights_non_entailment)],
    marker=dict(color='orange')
))

# Update layout to place the two charts vertically
fig.update_layout(
    title="Classification Head Weights for Entailment and Non-Entailment Classes",
    xaxis_title="Component Index",
    yaxis_title="Weight",
    barmode='relative',  # Ensures bars are positioned for easy comparison
    hovermode="x unified",  # Show all hover information at the same x position
    showlegend=True
)

# Separate subplots for better clarity
fig.update_yaxes(title_text="Entailment Weights", row=1, col=1)
fig.update_yaxes(title_text="Non-Entailment Weights", row=2, col=1)

# Display the figure
fig.show()

# Load the JSON data
with open('post_processing/results_greater-than_GPT2_t0.01585_renamed.json', 'r') as file:
    data = json.load(file)

nodes = data['nodes']
edges = data['edges']

# Initialize a PyGraphviz AGraph
G = pgv.AGraph(directed=True)

# Function to extract layer number from node name
def extract_layer_number(node):
    match = re.search(r'\d+', node)
    return int(match.group()) if match else float('inf')

# Add nodes to the graph with positions
layer_positions = {}
for node in nodes:
    layer = extract_layer_number(node)
    if layer not in layer_positions:
        layer_positions[layer] = len(layer_positions)
    pos = (layer_positions[layer], -layer)
    G.add_node(node, pos=f"{pos[0]},{pos[1]}!")

# Add edges to the graph
for start, end in edges:
    G.add_edge(start, end)

# Render the graph to a file
output_file = 'graph.png'
G.layout(prog='dot')
G.draw(output_file)

# Display the graph using matplotlib
img = plt.imread(output_file)
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.title('Directed Graph Visualization')
plt.show()
