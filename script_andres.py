import numpy as np
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate

# sorry to steal your huggingface token here, but I needed it as the dataset is private
login(token="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", add_to_git_credential=True)
dataset_name = "andres-vs/ruletaker-Att-Noneg-depth0"
# as you do sentence classification in this task, I used a classification model instead of an autoregressive one
model_name = "bert-base-uncased"
print("Loading dataset")

dataset = load_dataset(dataset_name)

def remove_special_tokens(example):
    example['input'] = example['input'].replace('[SEP]', ' ').replace('[CLS]', ' ')
    return {'text': example['input'], 'label': example['label']}

dataset = dataset.map(remove_special_tokens)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# this just prints some data entries as a sanity check
for i in range(10):
    print(dataset["train"][i])
    t = tokenizer(dataset["train"][i]["input"], truncation=True)
    print(len(t["input_ids"]))
    print(tokenizer.decode(t["input_ids"]))

print("Loaded tokenizer...")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


tokenized_datasets = dataset.map(tokenize_function, batched=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    logging_steps=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    max_grad_norm=1.0,
    learning_rate=1e-5,
    weight_decay=0.01,
    warmup_steps=100,
    num_train_epochs=20,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
