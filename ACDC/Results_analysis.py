import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the JSON data
with open('post_processing/acdc_results_text-entailment_BERT_depth0_t0.02_retrained.json', 'r') as file:
    data = json.load(file)['data']

text = data[0]['text']
y_values = data[0]['y']

# Initialize dictionaries to store the aggregated values for each attention head and MLP node
attention_head_impact = {}
mlp_node_impact = {}

# Iterate through the data to aggregate the values
for edge, y_value in zip(text, y_values):
    [start, end] = edge.split(' to ')
    if 'attn' in start:
        layer = re.search(r'blocks\.(\d+)\.', start).group(1)
        head = re.search(r'attn\.hook_result\[:, :, (\d+)\]', start).group(1)
        if (layer, head) not in attention_head_impact:
            attention_head_impact[(layer, head)] = 0
        attention_head_impact[(layer, head)] += y_value
    elif 'hook_mlp_out' in start:
        layer = re.search(r'blocks\.(\d+)\.', start).group(1)
        if layer not in mlp_node_impact:
            mlp_node_impact[layer] = 0
        mlp_node_impact[layer] += y_value

# Create a 2D array to store the heatmap values for attention heads
heatmap_data_attn = np.full((12, 12), np.nan)

# Fill the heatmap data with the attention head impacts
for (layer, head), impact in attention_head_impact.items():
    heatmap_data_attn[int(layer), int(head)] = impact

# Create the heatmap plot for attention heads
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data_attn, annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=.5, linecolor='grey', mask=np.isnan(heatmap_data_attn), cbar_kws={'label': 'Impact'})
plt.title('Attention Head Impact Heatmap')
plt.xlabel('Head Number')
plt.ylabel('Layer Number')
plt.xticks(np.arange(12) + 0.5, np.arange(12))
plt.yticks(np.arange(12) + 0.5, np.arange(12))
plt.show()

# Create a 2D array to store the heatmap values for MLP nodes
heatmap_data_mlp = np.full((12, 1), np.nan)

# Fill the heatmap data with the MLP node impacts
for layer, impact in mlp_node_impact.items():
    heatmap_data_mlp[int(layer), 0] = impact

# Create the heatmap plot for MLP nodes
plt.figure(figsize=(2, 10))
sns.heatmap(heatmap_data_mlp, annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=.5, linecolor='grey', mask=np.isnan(heatmap_data_mlp), cbar_kws={'label': 'Impact'})
plt.title('MLP Node Impact Heatmap')
plt.xlabel('MLP')
plt.ylabel('Layer Number')
plt.xticks([0.5], [''])
plt.yticks(np.arange(12) + 0.5, np.arange(12))
plt.show()