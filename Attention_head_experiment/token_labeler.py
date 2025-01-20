#!/usr/bin/env python3
import json
import sys
import re
from tqdm import tqdm
import os

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
    # sentence = sentence.replace(',', ' ,')
    # Then split on whitespace
    tokens = sentence.split()
    return tokens

def is_rule_sentence(sentence):
    """
    Heuristic to decide if sentence is a rule:
    - Contains 'If' or 'All' or possibly 'things are' pattern
    """
    # Lowercase check (but store original form of tokens)
    lower = sentence.lower()
    if 'if ' in lower or 'all ' in lower:
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
    - else we guess 'OBJECT' or 'PROPERTY' (you may refine this)
    """
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
    elif t_lower in ['someone', 'something', 'they', 'it', 'people', 'person']:
        return 'OBJECT_PLACEHOLDER'
    else:
        # You could try more sophisticated checks, e.g.:
        # if token matches a known set of object names (Harry, Bob, etc.)
        # or if token is capitalized -> maybe it's an OBJECT
        # Otherwise -> PROPERTY
        # But let's do a naive approach:
        if token[0].isupper():
            # Heuristic: uppercase initial => object (e.g. "Harry")
            return 'OBJECT'
        else:
            return 'PROPERTY'

def label_fact_tokens(tokens):
    """
    Label tokens in a fact sentence (like "Harry is big" or "Dave is not rough").
    We'll treat the entire sentence as a fact, 
    and guess object vs property around 'is' or 'are'.
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
                # If it's "not" => NEGATION, else object
                if t.lower() == 'not':
                    labeled.append((t, '-', 'NEGATION'))
                else:
                    # Heuristic: If capitalized, assume object. Else property.
                    if t[0].isupper():
                        labeled.append((t, '-', 'OBJECT'))
                    else:
                        labeled.append((t, '-', guess_semantic_label(t)))
            else:
                # If it's "not" => NEGATION, else property
                if t.lower() == 'not':
                    labeled.append((t, '-', 'NEGATION'))
                else:
                    labeled.append((t, '-', 'PROPERTY'))
    else:
        # fallback: no "is/are"? Just label everything as a property or object
        # Typically won't happen if it's truly a fact, but who knows.
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
    for s_idx, sent in enumerate(sentences):
        # Label the sentence
        token_labels = label_sentence(sent)
        for t_idx, (token, fact_rule, premise_consequence, semantic_label) in enumerate(token_labels):
            # Build row
            row = {
                'doc_id': doc_id,
                'q_id': 'context',
                'sent_idx': s_idx,
                'token_idx': t_idx,
                'token': token,
                'part': 'theory',
                'fact_rule': fact_rule,
                'premise_consequence': premise_consequence,
                'semantic_label': semantic_label
            }
            all_rows.append(row)
        # If you want to preserve the DOT as a token for each sentence:
        dot_row = {
            'doc_id': doc_id,
            'q_id': 'context',
            'sent_idx': s_idx,
            'token_idx': len(token_labels),
            'token': '.',
            'part': 'theory',
            'fact_rule': token_labels[-1][1] if token_labels else 'fact',
            'premise_consequence': token_labels[-1][2] if token_labels else '-',
            'semantic_label': 'DOT'
        }
        all_rows.append(dot_row)

    return all_rows

def label_query(doc_id, q_id, text):
    """
    Labels the 'query' text in the same manner.
    A query might be a single sentence like "Harry is not round." or
    it might contain multiple sentences. Usually it's just one, but let's handle multiple.
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
                'token': token,
                'part': 'query',
                'fact_rule': fact_rule,
                'premise_consequence': premise_consequence,
                'semantic_label': semantic_label
            }
            all_rows.append(row)
        # Add final DOT if desired
        dot_row = {
            'doc_id': doc_id,
            'q_id': q_id,
            'sent_idx': s_idx,
            'token_idx': len(token_labels),
            'token': '.',
            'part': 'query',
            'fact_rule': token_labels[-1][1] if token_labels else 'fact',
            'premise_consequence': token_labels[-1][2] if token_labels else '-',
            'semantic_label': 'DOT'
        }
        all_rows.append(dot_row)
    return all_rows

def main(input_file, output_file=None):
    """
    Main driver:
    1. Read each line from input_file as JSON.
    2. Parse the 'context' + 'questions'.
    3. Label each token in the context (theory) + queries.
    4. Write results to output_file as TSV.
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

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        # Write header
        header = [
            'doc_id', 'q_id', 'sent_idx', 'token_idx',
            'token', 'part', 'fact_rule',
            'premise_consequence', 'semantic_label'
        ]
        f_out.write('\t'.join(header) + '\n')

        for line in tqdm(f_in, desc="Processing lines"):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            doc_id = data.get('id', 'unknown_id')
            context_text = data.get('context', '')
            questions = data.get('questions', [])

            # Label context
            context_rows = label_context(doc_id, context_text)
            for r in context_rows:
                row_str = '\t'.join(str(r[col]) for col in header)
                f_out.write(row_str + '\n')

            # Label each question
            for q in questions:
                q_id = q.get('id', '')
                q_text = q.get('text', '')
                q_rows = label_query(doc_id, q_id, q_text)
                for r in q_rows:
                    row_str = '\t'.join(str(r[col]) for col in header)
                    f_out.write(row_str + '\n')


if __name__ == '__main__':
    # Example usage:
    # python label_script.py input.jsonl output.tsv

    if len(sys.argv) < 2:
        print("Usage: python label_script.py <input_file.jsonl> [<output_file.tsv>]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    main(input_file, output_file)
