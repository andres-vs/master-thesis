import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the JSON data
with open('results_acdc_text-entailment_BERT_0.02.json', 'r') as file:
    data = json.load(file)['data']

text = data[0]['text']
y_values = data[0]['y']

# Initialize a dictionary to store the aggregated values for each attention head
attention_head_impact = {}

# Iterate through the data to aggregate the values
for edge, y_value in zip(text, y_values):
    [start, end] = edge.split(' to ')
    if 'attn' in start:
        layer = re.search(r'blocks\.(\d+)\.', start).group(1)
        head = re.search(r'attn\.hook_result\[:, :, (\d+)\]', start).group(1)
        if (layer, head) not in attention_head_impact:
            attention_head_impact[(layer, head)] = 0
        attention_head_impact[(layer, head)] += y_value

# Rank the attention heads based on the aggregated values
ranked_attention_heads = sorted(attention_head_impact.items(), key=lambda x: x[1], reverse=True)

# Print the ranking
for rank, (head, impact) in enumerate(ranked_attention_heads, start=1):
    print(f"Rank {rank}: {head} with impact {impact}")

# Create a 2D array to store the heatmap values
heatmap_data = np.full((12, 12), np.nan)

# Fill the heatmap data with the attention head impacts
for (layer, head), impact in attention_head_impact.items():
    heatmap_data[int(layer), int(head)] = impact

# Create the heatmap plot
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=.5, linecolor='grey', mask=np.isnan(heatmap_data), cbar_kws={'label': 'Impact'})
plt.title('Attention Head Aggregated Impact Heatmap')
plt.xlabel('Head Number')
plt.ylabel('Layer Number')
plt.xticks(np.arange(12) + 0.5, np.arange(12))
plt.yticks(np.arange(12) + 0.5, np.arange(12))
plt.show()