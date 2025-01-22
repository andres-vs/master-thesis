import torch
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

def load_tsv_annotations(tsv_file, example_id=None):
    """
    Loads token annotations from test.tsv.
    Returns a list of (token, semantic_label) for the specified example_id.
    
    If your tsv has multiple examples (doc_id or q_id), you'll want to filter.
    We'll do a simple filter by doc_id = example_id, q_id in {context, or something}.
    """
    tokens = []
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # row has keys:
            # doc_id, q_id, sent_idx, token_idx, token, part, fact_rule, premise_consequence, semantic_label
            if example_id is not None:
                if row['doc_id'] != example_id:
                    continue
            # If you only want the "theory" part or also "query" part, 
            # you can refine here. For now, let's collect everything.
            
            token = row['token']
            sem_label = row['semantic_label']
            
            # Exclude special tokens like [CLS], [SEP], if they appear
            # No exclusion for now, but you can add a condition like:
            # if token in ['[CLS]', '[SEP]']:
            #     continue
            
            tokens.append((token, sem_label))
    
    return tokens

def build_label_index(labels_list):
    """
    Create a stable index for each label (e.g. OBJECT -> 0, PROPERTY -> 1, etc.)
    Return both the dict label2idx and the reversed idx2label.
    """
    unique_labels = sorted(set(labels_list))
    label2idx = {label: i for i, label in enumerate(unique_labels)}
    idx2label = {i: label for label, i in label2idx.items()}
    return label2idx, idx2label

def aggregate_attention_by_category(attn_matrix, token_sem_labels):
    """
    attn_matrix: shape [seq_len, seq_len], with attention[i, j] = attention from token i -> token j
    token_sem_labels: list of length seq_len with the semantic label for each token
    
    Returns a 2D np.array [n_labels x n_labels] with aggregated sums of attention.
    """
    # Collect all label types to build a consistent matrix
    all_labels = sorted(set(token_sem_labels))
    label2idx, idx2label = build_label_index(all_labels)
    n_labels = len(all_labels)
    
    # Initialize aggregator matrix
    agg_matrix = np.zeros((n_labels, n_labels))
    
    seq_len = len(token_sem_labels)
    for i in range(seq_len):
        label_i = token_sem_labels[i]
        idx_i = label2idx[label_i]
        for j in range(seq_len):
            label_j = token_sem_labels[j]
            idx_j = label2idx[label_j]
            # add attn value
            agg_matrix[idx_i, idx_j] += attn_matrix[i, j].item()
    
    return agg_matrix, all_labels

def plot_heatmap(matrix, labels, title='Category Attention'):
    """
    Simple heatmap plot using matplotlib.
    matrix: 2D numpy array [n_labels x n_labels]
    labels: list of label names in index order
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='Blues')
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
    
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def main(tsv_file, pt_file, example_idx=0, layer_idx=0, head_idx=1, doc_id_filter=None):
    """
    1) Load tokens + semantic labels from the TSV for one example (doc_id_filter).
    2) Load attention patterns from PT file.
    3) Retrieve attention for [layer_idx][head_idx] on example_idx.
    4) Discard any padding positions if needed (based on the tokens length).
    5) Aggregate the attention into label→label sums.
    6) Plot or print the heatmap.
    """
    # 1) Load token info from TSV
    tokens_and_labels = load_tsv_annotations(tsv_file, example_id=doc_id_filter)
    seq_len = len(tokens_and_labels)
    # e.g. tokens_and_labels = [('If', 'IF_KEYWORD'), ('something','OBJECT_PLACEHOLDER'), ...]
    
    # Just keep the semantic labels in the same order
    semantic_labels = [sl for (_, sl) in tokens_and_labels]
    
    # 2) Load attention
    patterns = torch.load(pt_file)  # This might be a list of length = number of examples
    
    # 3) Get the attention for the example of interest
    #    patterns[example_idx] = { layer_idx: { head_idx: attention_tensor, ...}, ... }
    example_obj = patterns[example_idx]
    
    # Some caution: example_obj[layer_idx] might be a dictionary with head_idx keys
    # If your data structure is nested differently, adapt accordingly:
    layer_dict = example_obj[layer_idx]  # { head_number: attention_matrix, ... }
    attn_matrix = layer_dict[head_idx]   # shape could be [seq_len, seq_len] or [1, seq_len, seq_len]
    
    # If it has a batch dimension or so, remove it if necessary
    # For example, if shape is [1, seq_len, seq_len], do attn_matrix = attn_matrix[0]
    if attn_matrix.dim() == 3 and attn_matrix.size(0) == 1:
        attn_matrix = attn_matrix[0]
    
    # 4) Discard padding positions if attention_matrix is bigger than seq_len
    #    We'll slice the top-left corner
    attn_matrix = attn_matrix[:seq_len, :seq_len]
    
    # 5) Aggregate
    agg_matrix, label_list = aggregate_attention_by_category(attn_matrix, semantic_labels)
    
    # 6) Plot
    plot_heatmap(agg_matrix, label_list, 
                 title=f"Layer {layer_idx}, Head {head_idx} Category Attention (Example {example_idx})")


if __name__ == "__main__":
    # Example usage:
    # python category_attention.py path/to/test.tsv path/to/attention_patterns.pt \
    #        [example_idx] [layer_idx] [head_idx] [doc_id_filter]
    
    # example_idx, layer_idx, head_idx are optional. doc_id_filter can be used if your test.tsv
    # has multiple doc_ids and you want to pick a specific one (e.g. "AttNoneg-D0-2135").
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python category_attention.py <test.tsv> <attention_patterns.pt> [example_idx=0] [layer_idx=0] [head_idx=1] [doc_id_filter=None]")
        sys.exit(1)
    
    tsv_file = sys.argv[1]
    pt_file = sys.argv[2]
    
    # Optional args
    example_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    layer_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    head_idx = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    doc_id_filter = sys.argv[6] if len(sys.argv) > 6 else None
    
    main(tsv_file, pt_file, example_idx, layer_idx, head_idx, doc_id_filter)
