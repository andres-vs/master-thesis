from matplotlib.colors import ListedColormap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load comments from CSV
def load_comments(file_path):
    return pd.read_csv(file_path)

# Identify layer-head pairs
def identify_layer_head_pairs(df, num_layers, num_heads):
    all_pairs = set((layer, head) for layer in range(num_layers) for head in range(num_heads))
    type1_pairs = set((row['Layer'], row['Head']) for _, row in df[df['Input_Type'] == 1].iterrows())
    type3_pairs = set((row['Layer'], row['Head']) for _, row in df[df['Input_Type'] == 3].iterrows())

    no_comments_pairs = all_pairs - type1_pairs
    type1_only_pairs = type1_pairs - type3_pairs
    both_types_pairs = type1_pairs & type3_pairs

    return no_comments_pairs, type1_only_pairs, both_types_pairs

# Create heatmap data
def create_heatmap_data(no_comments_pairs, type1_only_pairs, both_types_pairs, num_layers, num_heads):
    heatmap_data = []
    for layer in range(num_layers):
        row = []
        for head in range(num_heads):
            if (layer, head) in no_comments_pairs:
                row.append(0)  # No Comments
            elif (layer, head) in type1_only_pairs:
                row.append(1)  # Type 1 Only
            elif (layer, head) in both_types_pairs:
                row.append(2)  # Both Types
            # else:
            #     row.append(3)  # N/A
        heatmap_data.append(row)
    return heatmap_data

# Visualize heatmap using matplotlib
def visualize_heatmap(heatmap_data):
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(["white", "blue", "green"])
    ax = sns.heatmap(heatmap_data, annot=False, fmt='d', cmap=cmap, cbar=False, linewidths=.5, linecolor='grey')
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    plt.title('Function differentiation between input types')

    # Create a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Not important'),
        Patch(facecolor='blue', edgecolor='black', label='General importance'),
        Patch(facecolor='green', edgecolor='black', label='Differing functionalities'),
        # Patch(facecolor='red', edgecolor='black', label='N/A')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.show()

# Main function to run the workflow
def main(file_path):
    comments_df = load_comments(file_path)
    num_layers = 12
    num_heads = 12
    no_comments_pairs, type1_only_pairs, both_types_pairs = identify_layer_head_pairs(comments_df, num_layers, num_heads)
    heatmap_data = create_heatmap_data(no_comments_pairs, type1_only_pairs, both_types_pairs, num_layers, num_heads)
    visualize_heatmap(heatmap_data)

if __name__ == "__main__":
    main('comments_analysis_sorted.csv')