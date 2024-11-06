import numpy as np
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer
import sys

print("Starting set-up")
login()
# login(token="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", add_to_git_credential=True)

dataset_name = sys.argv[1] #"andres-vs/ruletaker-Att-Noneg-depth0"
model_name = "bert-base-uncased"

print("Loading dataset")
dataset = load_dataset(dataset_name)

print("Tokenizing dataset")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# helper functions
def remove_special_tokens(example):
    example['input'] = example['input'].replace('[SEP]', ' ').replace('[CLS]', ' ')
    return {'text': example['input'], 'label': example['label']}

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

# remove special tokens
dataset = dataset.map(remove_special_tokens)
# tokenize
tokenized_datasets = dataset.map(tokenize_function, batched=True)

print("Saving tokenized dataset")
tokenized_datasets.save_to_disk(f"$VSC_DATA/ruletaker_tokenized/{dataset_name.split('/ruletaker-')[1]}")