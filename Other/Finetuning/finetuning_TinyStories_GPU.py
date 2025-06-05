import numpy as np
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import wandb
import os

print("Starting set-up")
login(token="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", add_to_git_credential=True)

dataset_name = "andres-vs/ruletaker-Att-Noneg-depth0"
model_name = "roneneldan/TinyStories-8M"
tokenizer_name = "EleutherAI/gpt-neo-125M"

print("Loading dataset")
dataset = load_dataset(dataset_name)

print("Tokenizing dataset")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # add special token for padding
default_data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
print(tokenizer)
# helper functions
def remove_special_tokens(example):
    example['input'] = example['input'].replace('[CLS]', '')
    return {'text': example['input'], 'label': example['label']}

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")

# remove special tokens
dataset = dataset.map(remove_special_tokens)
# tokenize
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# set the right format
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
# # remove 'text' column
# tokenized_datasets = tokenized_datasets.remove_columns(["text"])

print("Loading model")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, pad_token_id=tokenizer.pad_token_id)
model.resize_token_embeddings(len(tokenizer)) # resize the token embeddings to match the tokenizer
metric = evaluate.load("accuracy")

print("Setting up training")
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]=f"{model_name.split('/')[1]}-finetuned_{dataset_name.split('/ruletaker-')[1]}"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir="test_BERT",
    evaluation_strategy="epoch",
    logging_steps=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    max_grad_norm=1.0,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=100,
    num_train_epochs=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)
print("Training")
trainer.train()
wandb.finish()