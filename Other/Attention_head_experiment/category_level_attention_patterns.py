import json
import torch
import csv
import sys
import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

ALL_POSSIBLE_LABELS = sorted(['ALL_KEYWORD', 'CLS', 'CONJUNCTION', 'DOT', 'IF_KEYWORD', 'OBJECT', 'NEGATION', 'OBJECT_PLACEHOLDER', 'PROPERTY', 'PUNCT', 'SEP', 'THEN_KEYWORD', 'VERB_IS'])

def load_annotations_grouped(tsv_file, doc_id_filter=None, example_indices=None):
    """
    Reads the entire test.tsv and groups tokens by (doc_id, q_id).
    This version uses 'total_idx' to order tokens, ignoring sent_idx/token_idx.
    Optionally filters by a list or range of example indices.
    Returns:
      grouped[(doc_id, q_id)] = list of (total_idx, token, semantic_label) 
                               sorted by total_idx.
    """
    grouped = {}
    example_counter = 0
    with open(tsv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if example_indices is not None and example_counter not in example_indices:
                example_counter += 1
                continue
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
    doc_ids = []
    seen_doc_ids = set()
    for k in grouped.keys():
        doc_id = k[0]
        if doc_id not in seen_doc_ids:
            doc_ids.append(doc_id)
            seen_doc_ids.add(doc_id)
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
        # query_ids.sort()

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
    all_labels = ALL_POSSIBLE_LABELS#sorted(set(token_sem_labels))
    label2idx, idx2label = build_label_index(all_labels)
    n_labels = len(all_labels)

    agg_matrix = np.zeros((n_labels, n_labels))
    seq_len = len(token_sem_labels)

    # plt.figure(figsize=(8,6))
    # ax = sns.heatmap(attn_matrix,
    #                  cmap='Blues',
    #                  annot=False)  # Turn off numeric annotations
    # plt.show()
    for i in range(seq_len):
        label_i = token_sem_labels[i]
        row_i = label2idx[label_i]
        for j in range(seq_len):
            label_j = token_sem_labels[j]
            col_j = label2idx[label_j]
            agg_matrix[row_i, col_j] += attn_matrix[i, j].item()

    return agg_matrix, all_labels

def plot_heatmap(matrix, labels, save_path, title="Category Attention"):
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
    if save_path:
        plt.savefig(save_path + f"/{title}.png")
    else:
        plt.show()
    plt.close()

def main(tsv_file, pt_file, save_path=None, avg_examples=False, example_indices=[0], layer_idx=0, head_idx=0, doc_id_filter=None, circuit_file=None):
    """
    1) Load & group tokens from test.tsv based on total_idx for specified example_indices.
    2) Build examples by combining (doc_id, 'context') + (doc_id, query_id).
    3) Select the specified example_indices from that list.
    4) Load attention_patterns.pt, get [layer_idx][head_idx] for the selected example_indices.
    5) Slice the attention to the correct length, ignoring padding.
    6) Aggregate by category, then plot via seaborn.
    """
    # 1) Load grouped
    grouped = load_annotations_grouped(tsv_file, example_indices=example_indices, doc_id_filter=doc_id_filter)

    # 2) Build examples
    all_examples = build_examples_list(grouped)
    selected_examples = []
    for idx in example_indices:
        if idx >= len(all_examples):
            print(f"Error: example_idx={idx} is out of range (only {len(all_examples)} examples).")
            continue
        selected_examples.append(all_examples[idx])


    # 4) Load attention
    patterns = torch.load(pt_file)  # a list of dicts

    if circuit_file:
        with open(circuit_file, 'r') as f:
            circuit_data = json.load(f)
            attention_heads = [node for node in circuit_data['nodes'] if node.startswith('a')]
            for head in tqdm(attention_heads, total=len(attention_heads), desc="Processing attention heads"):
                layer_idx, head_idx = map(int, head[1:].split('h'))
                process_attention_head(layer_idx, head_idx, example_indices, selected_examples, patterns, avg_examples, save_path)
    else:
        process_attention_head(layer_idx, head_idx, example_indices, selected_examples, patterns, avg_examples, save_path)

def process_attention_head(layer_idx, head_idx, example_indices, selected_examples, patterns, avg_examples, save_path):
    aggregated_matrices = []
    for example_idx, (doc_qid, combined_tokens) in zip(example_indices, selected_examples):
        semantic_labels = [x[1] for x in combined_tokens]  # extract just the label
        seq_len = len(combined_tokens)

        # print(f"Using example_idx={example_idx} with doc_id={doc_id}, q_id={q_id}, seq_len={seq_len}")

        
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
            print(f"Error: head_idx={head_idx} not in layer {layer_idx} of example {example_idx}. Keys: {layer_data.keys()}")
            return

        # 5) Slice the attention to the correct length, ignoring padding
        attn_matrix = attn_matrix[:seq_len, :seq_len]

        # 6) Aggregate by category
        agg_matrix, labels = aggregate_attention_by_category(attn_matrix, semantic_labels)
        aggregated_matrices.append(agg_matrix)
    
    if avg_examples and aggregated_matrices:
        avg_matrix = np.mean(aggregated_matrices, axis=0)
        examples = example_indices if len(example_indices) < 5 else f"{example_indices[0]}-{example_indices[-1]}"
        plot_heatmap(avg_matrix, ALL_POSSIBLE_LABELS, save_path, title=f"avg_of_examples{examples}--head{layer_idx}.{head_idx}")
    else:
        for idx, matrix in enumerate(aggregated_matrices):
            plot_heatmap(matrix, ALL_POSSIBLE_LABELS, save_path, title=f"example{examples[idx]}--head{layer_idx}.{head_idx}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Category Attention Visualization")
    parser.add_argument("tsv_file", type=str, help="Path to the test.tsv file")
    parser.add_argument("pt_file", type=str, help="Path to the attention_patterns.pt file")
    parser.add_argument("--avg_examples", action="store_true", default=False, help="Average attention across all examples")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the plot")
    parser.add_argument("--example_indices", type=str, default="0", help="Comma-separated list or range of example indices (e.g., '0,1,2' or '0-2')")
    parser.add_argument("--layer_idx", type=int, default=0, help="Layer index to visualize")
    parser.add_argument("--head_idx", type=int, default=0, help="Head index to visualize")
    parser.add_argument("--doc_id_filter", type=str, default=None, help="Optional doc_id filter")
    parser.add_argument("--circuit_file", type=str, help="Path to the circuit description JSON file.")

    args = parser.parse_args()

    # Parse example_indices
    if '-' in args.example_indices:
        start, end = map(int, args.example_indices.split('-'))
        example_indices = list(range(start, end + 1))
    else:
        example_indices = list(map(int, args.example_indices.split(',')))

    main(args.tsv_file, args.pt_file, args.save_path, args.avg_examples, example_indices, args.layer_idx, args.head_idx, args.doc_id_filter, args.circuit_file)
