import torch
import csv
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def load_annotations_grouped(tsv_file, doc_id_filter=None):
    """
    Reads the entire test.tsv and groups tokens by (doc_id, q_id).
    This version uses 'total_idx' to order tokens, ignoring sent_idx/token_idx.
    
    Skips lines where token is [CLS] or [SEP].
    Returns:
      grouped[(doc_id, q_id)] = list of (total_idx, token, semantic_label) 
                               sorted by total_idx.
    """
    grouped = {}
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            doc_id = row['doc_id']
            q_id = row['q_id']
            token = row['token']

            # Optionally filter by doc_id
            if doc_id_filter and doc_id != doc_id_filter:
                continue
            
            # Skip [CLS] and [SEP] tokens
            # if token in ['[CLS]', '[SEP]']:
            #     continue
            
            # Parse total_idx
            # Some lines might have None for token_idx, but total_idx should be valid for normal tokens
            ti_str = row.get('total_idx', None)
            if ti_str is None or ti_str == 'None':
                # If there's truly no total_idx, skip (or handle differently if you prefer)
                continue
            total_idx = int(ti_str)

            sem_label = row['semantic_label']
            
            # Insert into dict
            grouped.setdefault((doc_id, q_id), []).append((total_idx, token, sem_label))

    # Now sort each list by total_idx
    for key in grouped.keys():
        grouped[key].sort(key=lambda x: x[0])  # x is (total_idx, token, sem_label)
    return grouped

def build_examples_list(grouped):
    """
    Given the grouped dict of (doc_id, q_id) -> [(total_idx, token, semantic_label), ...],
    build a list of "examples" for each (doc_id, query_id) where query_id != 'context'.

    Each "example" is:
      ( (doc_id, q_id), combined_tokens )
    where combined_tokens is a list of (token, semantic_label)
    formed by *concatenating* the doc_id's context tokens + that query's tokens.

    The order in the returned list should match how you want to align with attention_patterns.pt:
    - We simply sort queries alphanumerically by q_id
    - If you need a different order, adapt here.
    """
    all_examples = []

    # Collect all doc_ids
    doc_ids = set(k[0] for k in grouped.keys())
    for doc_id in doc_ids:
        # Theory key
        context_key = (doc_id, 'context')
        if context_key not in grouped:
            # Possibly no context for this doc_id
            continue
        # Extract context tokens (sorted by total_idx)
        context_data = grouped[context_key]
        # Convert to list of (token, sem_label) ignoring total_idx
        context_tokens = [(t[1], t[2]) for t in context_data]

        # Find queries for this doc_id
        query_ids = [k[1] for k in grouped.keys() 
                     if k[0] == doc_id and k[1] != 'context']
        query_ids.sort()

        for qid in query_ids:
            query_data = grouped[(doc_id, qid)]
            query_tokens = [(t[1], t[2]) for t in query_data]
            # Combine
            combined_tokens = context_tokens + query_tokens
            # Make an example
            all_examples.append(((doc_id, qid), combined_tokens))

    return all_examples

def build_label_index(labels_list):
    """
    Create an index for each label so we can build a category->category matrix.
    """
    unique_labels = sorted(set(labels_list))
    label2idx = {label: i for i, label in enumerate(unique_labels)}
    idx2label = {i: label for label, i in label2idx.items()}
    return label2idx, idx2label

def aggregate_attention_by_category(attn_matrix, token_sem_labels):
    """
    attn_matrix: [seq_len, seq_len] attention from token i -> token j
    token_sem_labels: list[str] of length seq_len with semantic labels

    Returns (agg_matrix, label_list)
      - agg_matrix is [n_labels x n_labels] with sums
      - label_list is the sorted list of label names
    """
    all_labels = sorted(set(token_sem_labels))
    label2idx, idx2label = build_label_index(all_labels)
    n_labels = len(all_labels)

    agg_matrix = np.zeros((n_labels, n_labels))
    seq_len = len(token_sem_labels)

    for i in range(seq_len):
        label_i = token_sem_labels[i]
        row_i = label2idx[label_i]
        for j in range(seq_len):
            label_j = token_sem_labels[j]
            col_j = label2idx[label_j]
            agg_matrix[row_i, col_j] += attn_matrix[i, j].item()

    return agg_matrix, all_labels

def plot_heatmap(matrix, labels, title="Category Attention"):
    """
    Plot a heatmap using seaborn. No numeric annotations, just color.
    """
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(matrix,
                     xticklabels=labels,
                     yticklabels=labels,
                     cmap='Blues',
                     annot=False)  # Turn off numeric annotations
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def main(tsv_file, pt_file, example_idx=0, layer_idx=0, head_idx=0, doc_id_filter=None):
    """
    1) Load & group tokens from test.tsv based on total_idx (ignoring [CLS] / [SEP]).
    2) Build examples by combining (doc_id, 'context') + (doc_id, query_id).
    3) Select the example_idx-th from that list.
    4) Load attention_patterns.pt, get [layer_idx][head_idx] for that example_idx.
    5) Slice the attention to the correct length, ignoring padding.
    6) Aggregate by category, then plot via seaborn.
    """
    # 1) Load grouped
    grouped = load_annotations_grouped(tsv_file, doc_id_filter=doc_id_filter)

    # 2) Build examples
    all_examples = build_examples_list(grouped)
    if example_idx >= len(all_examples):
        print(f"Error: example_idx={example_idx} is out of range (only {len(all_examples)} examples).")
        return

    # 3) Pick the example
    (doc_qid, combined_tokens) = all_examples[example_idx]
    doc_id, q_id = doc_qid
    semantic_labels = [x[1] for x in combined_tokens]  # extract just the label
    seq_len = len(combined_tokens)

    print(f"Using example_idx={example_idx} => doc_id={doc_id}, q_id={q_id}, seq_len={seq_len}")

    # 4) Load attention
    patterns = torch.load(pt_file)  # a list of dicts
    if example_idx >= len(patterns):
        print(f"Error: The attention_patterns.pt does not have index {example_idx}.")
        return

    example_data = patterns[example_idx]
    layer_data = example_data.get(layer_idx, None)
    if layer_data is None:
        print(f"Error: layer_idx={layer_idx} not in example {example_idx}. Keys: {example_data.keys()}")
        return
    attn_matrix = layer_data.get(head_idx, None)
    if attn_matrix is None:
        print(f"Error: head_idx={head_idx} not found in layer {layer_idx}. Available heads: {layer_data.keys()}")
        return

    # If shape is [1, seq_len, seq_len], remove batch dim
    if attn_matrix.dim() == 3 and attn_matrix.size(0) == 1:
        attn_matrix = attn_matrix[0]

    # 5) Slice to remove padding
    attn_matrix = attn_matrix[:seq_len, :seq_len]

    # 6) Aggregate & plot
    agg_matrix, label_list = aggregate_attention_by_category(attn_matrix, semantic_labels)
    plot_heatmap(agg_matrix, label_list,
                 title=f"Layer={layer_idx}, Head={head_idx}\nExample={example_idx}  ({doc_id}, {q_id})")

if __name__ == "__main__":
    """
    Example usage:
      python category_attention.py test.tsv attention_patterns.pt 0 0 0 AttNoneg-D0-2135
      ^                 ^                ^ ^ ^ ^  
      |                 |                | | | `-- doc_id_filter
      |                 |                | | `---- head_idx
      |                 |                | `------ layer_idx
      |                 |                `-------- example_idx
      |                 `------------------------- path to attention_patterns.pt
      `------------------------------------------- path to test.tsv
    """
    if len(sys.argv) < 3:
        print("Usage: python category_attention.py <test.tsv> <attention_patterns.pt> [example_idx=0] [layer_idx=0] [head_idx=0] [doc_id_filter=None]")
        sys.exit(1)

    tsv_file = sys.argv[1]
    pt_file = sys.argv[2]
    example_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    layer_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    head_idx = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    doc_id_filter = sys.argv[6] if len(sys.argv) > 6 else None

    main(tsv_file, pt_file, example_idx, layer_idx, head_idx, doc_id_filter)
