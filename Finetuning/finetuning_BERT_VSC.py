import numpy as np
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import wandb
import os

print("Starting set-up")
# login()
login(token="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", add_to_git_credential=True)

dataset_name = "andres-vs/ruletaker-Att-Noneg-depth1"
model_name = "bert-base-uncased"

print("Loading dataset")
dataset = load_dataset(dataset_name)

print("Tokenizing dataset")
tokenizer = AutoTokenizer.from_pretrained(model_name)

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

print("Loading model")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
metric = evaluate.load("accuracy")

print("Setting up training")
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]=f"{model_name}-finetuned_{dataset_name.split('/ruletaker-')[1]}"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned_{dataset_name.split('/ruletaker-')[1]}",
    evaluation_strategy="epoch",
    logging_steps=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    max_grad_norm=1.0,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=100,
    num_train_epochs=20,
    # fp16=True,
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