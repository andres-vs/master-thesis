#!/usr/bin/env python3
import json
import sys
import re
from tqdm import tqdm
import os
import argparse

def split_into_sentences(text):
    """
    Splits text by '.' to get sentences, strips whitespace,
    and returns a list of non-empty sentences.
    """
    # Split on '.' and strip
    raw_sentences = text.split('.')
    # Clean up and remove empty
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences

def tokenize_sentence(sentence):
    """
    Simple tokenizer splitting on spaces and keeping commas, etc.
    You may refine this to handle punctuation more precisely or
    to replicate BERT-like tokenization. For now, we do a naive split.
    """
    # If you want to split out commas separately, you could do something like:
    sentence = sentence.replace(',', ' ,')
    # Then split on whitespace
    tokens = sentence.split()
    return tokens

def is_rule_sentence(sentence):
    """
    Heuristic to decide if sentence is a rule:
    - Contains 'If' or 'All'
    - Contains 'things are' pattern (like "Red things are big")
    """
    # Lowercase check (but store original form of tokens)
    lower = sentence.lower()
    if 'if ' in lower or 'all ' in lower:
        return True
    
    # Check for "[Property] things are" pattern
    if ' things are ' in lower or lower.startswith('things are '):
        return True
    
    # Check for patterns where a property (possibly capitalized) starts a sentence
    # followed by "are" - like "Red things are kind" or "Quiet people are nice"
    tokens = tokenize_sentence(lower)
    if len(tokens) >= 3:
        known_properties = {"red", "blue", "green", "kind", "nice", "big", "cold", "young", 
                           "round", "rough", "white", "smart", "quiet", "furry"}
        # Check if first word is a known property and 'are' appears in the sentence
        if tokens[0] in known_properties and 'are' in tokens:
            return True
    
    return False

def label_rule_tokens(tokens):
    """
    Given a list of tokens in a rule, we label which are part
    of the premise vs. consequence (for 'If X then Y' type),
    or handle the universal form like 'All X things are Y'.
    Returns a list of tuples (token, premise_or_consequence_label, semantic_label).
    """
    # Join them to find the structure, ignoring case for detection:
    joined_lower = ' '.join(t.lower() for t in tokens)

    # Detect "If ... then ..." format
    if 'if ' in joined_lower and ' then ' in joined_lower:
        # We'll split on the 'then' region
        try:
            # Find indices for 'If' and 'then'
            if_idx = None
            then_idx = None
            # We do a single pass to find 'if' and 'then' (case-insensitive)
            for i, t in enumerate(tokens):
                if t.lower() == 'if':
                    if_idx = i
                if t.lower() == 'then':
                    then_idx = i

            # If we found both, apply premise vs. consequence
            # Everything from if_idx+1 up to then_idx is premise
            # Everything from then_idx+1 to end is consequence
            labeled = []
            for i, t in enumerate(tokens):
                # Mark 'If' and 'then' with special semantic labels
                if i == if_idx:
                    labeled.append((t, 'premise (marker)', 'IF_KEYWORD'))
                elif i == then_idx:
                    labeled.append((t, 'consequence (marker)', 'THEN_KEYWORD'))
                else:
                    # Check if we are in premise or consequence
                    if if_idx is not None and then_idx is not None:
                        if i > if_idx and i < then_idx:
                            labeled.append((t, 'premise', guess_semantic_label(t)))
                        elif i > then_idx:
                            labeled.append((t, 'consequence', guess_semantic_label(t)))
                        else:
                            # If something is outside, just mark with '-'
                            labeled.append((t, '-', guess_semantic_label(t)))
                    else:
                        labeled.append((t, '-', guess_semantic_label(t)))
            return labeled
        except:
            pass

    # Detect "All X things are Y" or "Quiet things are kind." style
    # We'll consider the "are" as a split point
    if 'all ' in joined_lower or ' things are ' in joined_lower:
        # find index of 'are' if it exists
        are_idx = None
        for i, t in enumerate(tokens):
            if t.lower() == 'are':
                are_idx = i
                break

        # If we found an "are", treat the left side as premise, right as consequence
        labeled = []
        if are_idx is not None:
            # Mark any 'All' token specially
            # Then everything up to 'are_idx' is premise, everything after is consequence
            for i, t in enumerate(tokens):
                if t.lower() == 'all':
                    labeled.append((t, 'premise (marker)', 'ALL_KEYWORD'))
                elif i < are_idx:
                    labeled.append((t, 'premise', guess_semantic_label(t)))
                elif i == are_idx:
                    # 'are'
                    labeled.append((t, 'consequence (marker)', 'VERB_IS'))
                else:
                    labeled.append((t, 'consequence', guess_semantic_label(t)))
            return labeled

    # Fallback: we could not parse structure well; treat entire sentence as a rule
    return [(t, '-', guess_semantic_label(t)) for t in tokens]

def guess_semantic_label(token):
    """
    Very rough guess of semantic label:
    - 'is' -> VERB_IS
    - 'are' -> VERB_IS
    - 'not' -> NEGATION
    - 'and' -> CONJUNCTION
    - 'or' -> CONJUNCTION
    - simple placeholders 'someone', 'something', 'they', 'it' -> OBJECT_PLACEHOLDER
    - punctuation
    - else we check against known OBJECT and PROPERTY lists
    """
    # Define known objects and properties
    known_objects = {"anne", "bob", "charlie", "dave", "erin", "fiona", "gary", "harry"}
    known_properties = {"red", "blue", "green", "kind", "nice", "big", "cold", "young", 
                        "round", "rough", "white", "smart", "quiet", "furry"}
    
    t_lower = token.lower()

    if t_lower in ['is', 'are']:
        return 'VERB_IS'
    elif t_lower in ['not']:
        return 'NEGATION'
    elif t_lower in ['and', 'or']:
        return 'CONJUNCTION'
    elif t_lower in ['if']:
        return 'IF_KEYWORD'
    elif t_lower in ['then']:
        return 'THEN_KEYWORD'
    elif t_lower in ['all']:
        return 'ALL_KEYWORD'
    elif t_lower in ['.', '?', ',', ';']:
        return 'PUNCT'
    elif t_lower in ['someone', 'something', 'they', 'it', 'people', 'person', 'things']:
        return 'OBJECT_PLACEHOLDER'
    # Check against known objects and properties
    elif t_lower in known_objects:
        return 'OBJECT'
    elif t_lower in known_properties:
        return 'PROPERTY'
    else:
        # Fallback to the original heuristic for unknown tokens
        if token[0].isupper():
            return 'OBJECT'
        else:
            return 'PROPERTY'

def label_fact_tokens(tokens):
    """
    Label tokens in a fact sentence (like "Harry is big" or "Dave is not rough").
    We'll treat the entire sentence as a fact, 
    and guess object vs. property around 'is' or 'are'.
    """
    labeled = []
    # We can find "is" / "are" to identify object vs. property
    # For a more robust approach, you'd parse grammatically, but let's keep it simple
    if_indices = [i for i, t in enumerate(tokens) if t.lower() in ['is', 'are']]
    if if_indices:
        # We'll just handle the first "is/are" for a single fact
        cop_idx = if_indices[0]
        for i, t in enumerate(tokens):
            # If it's the copula itself (is/are), label as VERB_IS
            if i == cop_idx:
                labeled.append((t, '-', 'VERB_IS'))
            # If it's before, guess object
            elif i < cop_idx:
                # Special case for "not", otherwise use guess_semantic_label
                if t.lower() == 'not':
                    labeled.append((t, '-', 'NEGATION'))
                else:
                    labeled.append((t, '-', guess_semantic_label(t)))
            else:
                # If it's after the copula, it could be "not" or a property
                if t.lower() == 'not':
                    labeled.append((t, '-', 'NEGATION'))
                else:
                    labeled.append((t, '-', guess_semantic_label(t)))
    else:
        # fallback: no "is/are"? Just use guess_semantic_label for everything
        labeled = [(t, '-', guess_semantic_label(t)) for t in tokens]

    return labeled

def label_sentence(sentence):
    """
    Decide if it's a fact or a rule, then label tokens accordingly.
    Returns a list of (token, fact_rule, premise_consequence, semantic_label).
    """
    # 1) Tokenize
    tokens = tokenize_sentence(sentence)

    # 2) Fact or rule
    if is_rule_sentence(sentence):
        fact_rule = 'rule'
        # Label tokens for premise/consequence, etc.
        labeled = label_rule_tokens(tokens)
    else:
        fact_rule = 'fact'
        # Label tokens in a simpler way for facts
        tmp = label_fact_tokens(tokens)
        # For facts, we don’t really have premise/consequence, so we set both to '-'
        labeled = [(t[0], fact_rule, '-', t[2]) for t in tmp]

    # If we labeled it as a rule, label_rule_tokens returns (token, premise/..., semantic_label).
    # We need to unify them into the format: (token, fact_rule, premise_consequence, semantic_label).
    if fact_rule == 'rule':
        # labeled is list of (token, premise_or_consequence, semantic_label)
        output = []
        for t, pc, sl in labeled:
            output.append((t, fact_rule, pc, sl))
        return output
    else:
        return labeled

def label_context(doc_id, text):
    """
    Labels the entire 'context' (theory) as a series of sentences,
    returns a list of dictionaries with all the relevant info.
    """
    sentences = split_into_sentences(text)
    all_rows = []
    
    # Track total token index across context (and query, if extended similarly)
    total_idx = 0

    # Insert [CLS] token at the beginning of the context
    all_rows.append({
        'doc_id': doc_id,
        'q_id': 'context',
        'sent_idx': -1,  # or any sentinel value
        'token_idx': 0,
        'total_idx': total_idx,
        'token': '[CLS]',
                'part': 'theory',
        'fact_rule': None,
        'premise_consequence': None,
        'semantic_label': "CLS"
    })
    total_idx += 1

    for s_idx, sent in enumerate(sentences):
        # Label the sentence
        token_labels = label_sentence(sent)
        for t_idx, (token, fact_rule, premise_consequence, semantic_label) in enumerate(token_labels):
            # Build row with total token index
            row = {
                'doc_id': doc_id,
                'q_id': 'context',
                'sent_idx': s_idx,
                'token_idx': t_idx,
                'total_idx': total_idx,
                'token': token,
                'part': 'theory',
                'fact_rule': fact_rule,
                'premise_consequence': premise_consequence,
                'semantic_label': semantic_label
            }
            all_rows.append(row)
            total_idx += 1
        # Add final DOT for every sentence if desired
        dot_row = {
            'doc_id': doc_id,
            'q_id': 'context',
            'sent_idx': s_idx,
            'token_idx': len(token_labels),
            'total_idx': total_idx,
            'token': '.',
            'part': 'theory',
            'fact_rule': token_labels[-1][1] if token_labels else 'fact',
            'premise_consequence': None,
            'semantic_label': 'DOT'
        }
        all_rows.append(dot_row)
        total_idx += 1

    # Insert [SEP] token at the end of the context
    all_rows.append({
        'doc_id': doc_id,
        'q_id': 'context',
        'sent_idx': -1,  # or any sentinel value
        'token_idx': None,
        'total_idx': total_idx,
        'token': '[SEP]',
        'part': 'theory',
        'fact_rule': None,
        'premise_consequence': None,
        'semantic_label': "SEP"
    })
    total_idx += 1

    return all_rows, total_idx

def label_query(doc_id, q_id, text, total_idx, label=None, qdep=None, strategy=None):
    """
    Labels the 'query' text in the same manner.
    A query might be a single sentence like "Harry is not round." or
    it might contain multiple sentences. Usually it's just one, but let's handle multiple.
    
    Additional metadata parameters:
    - label: The label/answer of the query
    - qdep: The reasoning depth (QDep) of the query
    - strategy: The proof strategy used for the query
    """
    sentences = split_into_sentences(text)
    all_rows = []
    for s_idx, sent in enumerate(sentences):
        token_labels = label_sentence(sent)
        for t_idx, (token, fact_rule, premise_consequence, semantic_label) in enumerate(token_labels):
            row = {
                'doc_id': doc_id,
                'q_id': q_id,
                'sent_idx': s_idx,
                'token_idx': t_idx,
                'total_idx': total_idx,
                'token': token,
                'part': 'query',
                'fact_rule': fact_rule,
                'premise_consequence': premise_consequence,
                'semantic_label': semantic_label,
                'label': label,
                'reasoning_depth': qdep,
                'proof_strategy': strategy
            }
            all_rows.append(row)
            total_idx += 1
        # Add final DOT if desired
        dot_row = {
            'doc_id': doc_id,
            'q_id': q_id,
            'sent_idx': s_idx,
            'token_idx': len(token_labels),
            'total_idx': total_idx,
            'token': '.',
            'part': 'query',
            'fact_rule': token_labels[-1][1] if token_labels else 'fact',
            'premise_consequence': None,
            'semantic_label': 'DOT',
            'label': label,
            'reasoning_depth': qdep,
            'proof_strategy': strategy
        }
        all_rows.append(dot_row)
        total_idx += 1

        # Insert [SEP] token at the end of the query
        all_rows.append({
            'doc_id': doc_id,
            'q_id': q_id,
            'sent_idx': -1,  # or any sentinel value
            'token_idx': None,
            'total_idx': total_idx,
            'token': '[SEP]',
            'part': 'query',
            'fact_rule': None,
            'premise_consequence': None,
            'semantic_label': "SEP",
            'label': label,
            'reasoning_depth': qdep,
            'proof_strategy': strategy
        })
        total_idx += 1
    return all_rows

def main(input_file, output_file=None, rd_filter=None, ps_filter=None, only_attributes=False, no_negatives=False):
    """
    Main driver:
    1. Read each line from input_file as JSON.
    2. Parse the 'context' + 'questions'.
    3. Label each token in the context (theory) + queries.
    4. Write results to output_file as TSV.
    5. Apply filters for reasoning depth, proof strategy, and document types.
    
    Parameters:
    - input_file: Input JSONL file path
    - output_file: Output TSV file path (optional)
    - rd_filter: Filter by reasoning depth (QDep)
    - ps_filter: Filter by proof strategy
    - only_attributes: Filter for attribute documents ("Att" prefix) vs relation documents ("Rel" prefix)
    - no_negatives: Filter for "Noneg" vs "Neg" in document IDs
    """
    # Create the output directory if it doesn't exist
    input_dir = os.path.dirname(input_file)
    output_dir = os.path.join(input_dir, "labeled_tokens")
    os.makedirs(output_dir, exist_ok=True)

    # Set the output file name
    if output_file is None:
        input_filename = os.path.basename(input_file)
        output_filename = f"{os.path.splitext(input_filename)[0]}_tokens_labeled.tsv"
        output_file = os.path.join(output_dir, output_filename)

    # Determine document prefix filter based on arguments
    doc_prefix = None
    if only_attributes or no_negatives:
        prefix_part = "Att" if only_attributes else "Rel"
        neg_part = "Noneg" if no_negatives else "Neg"
        doc_prefix = f"{prefix_part}{neg_part}"
        print(f"Filtering documents with prefix: {doc_prefix}")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        # Write header
        header = [
            'doc_id', 'q_id', 'sent_idx', 'token_idx', 'total_idx',
            'token', 'part', 'fact_rule',
            'premise_consequence', 'semantic_label',
            'label', 'reasoning_depth', 'proof_strategy'
        ]
        f_out.write('\t'.join(header) + '\n')

        for line in tqdm(f_in, desc="Processing lines"):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            doc_id = data.get('id', 'unknown_id')
            
            # Apply document ID prefix filter if specified
            if doc_prefix and not doc_id.startswith(doc_prefix):
                continue
                
            context_text = data.get('context', '')
            questions = data.get('questions', [])
            
            # Apply filters - check if any question matches the filter criteria
            filtered_questions = []
            for q in questions:
                q_meta = q.get('meta', {})
                q_qdep = q_meta.get('QDep', None)
                q_strategy = q_meta.get('strategy', None)
                
                # Apply filters if specified
                keep_question = True
                if rd_filter is not None and str(q_qdep) != str(rd_filter):
                    keep_question = False
                if ps_filter is not None and q_strategy != ps_filter:
                    keep_question = False
                
                if keep_question:
                    filtered_questions.append(q)
            
            # Skip this context if no questions match the filters
            if (rd_filter is not None or ps_filter is not None) and not filtered_questions:
                continue
                
            # Label context only if we'll process at least one question
            context_rows, total_idx = label_context(doc_id, context_text)
            for r in context_rows:
                # Add None values for the new columns
                r['label'] = None
                r['reasoning_depth'] = None
                r['proof_strategy'] = None
                row_str = '\t'.join(str(r[col]) for col in header)
                f_out.write(row_str + '\n')

            # Label each question that passed the filters
            for q in filtered_questions if filtered_questions else questions:
                q_id = int(q.get('id', '').split("-")[-1])-1
                q_text = q.get('text', '')
                
                # Extract metadata
                q_label = q.get('label', None)
                q_meta = q.get('meta', {})
                q_qdep = q_meta.get('QDep', None)
                q_strategy = q_meta.get('strategy', None)
                
                q_rows = label_query(doc_id, q_id, q_text, total_idx, 
                                     label=q_label, qdep=q_qdep, strategy=q_strategy)
                for r in q_rows:
                    row_str = '\t'.join(str(r[col]) for col in header)
                    f_out.write(row_str + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Label tokens in context and queries with semantic information")
    parser.add_argument('input_file', help="Input JSONL file")
    parser.add_argument('output_file', nargs='?', default=None, help="Output TSV file (optional)")
    parser.add_argument('--RD', type=str, help="Filter by reasoning depth (QDep)", default=None)
    parser.add_argument('--PS', type=str, help="Filter by proof strategy", default=None)
    parser.add_argument('--only_attributes', action='store_true', help="Filter for attribute documents (Att prefix) vs relation documents (Rel prefix)")
    parser.add_argument('--no_negatives', action='store_true', help="Filter for documents with 'Noneg' vs 'Neg' in ID")
    
    args = parser.parse_args()
    main(args.input_file, args.output_file, rd_filter=args.RD, ps_filter=args.PS, 
         only_attributes=args.only_attributes, no_negatives=args.no_negatives)
