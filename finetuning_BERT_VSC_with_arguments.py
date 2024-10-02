import argparse
import subprocess
import numpy as np
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import wandb
import os
import torch
# import accelerate

def main():
    # print("Torch Version:", torch.__version__)
    print("torch.cuda.is_available():", torch.cuda.is_available())

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=20, help="Number of training epochs.")
    args = parser.parse_args()

    print("Starting set-up")
    # login()
    login(token="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", add_to_git_credential=True)

    dataset_name = args.dataset_name
    model_name = args.model_name

    print("Loading dataset")
    dataset = load_dataset(dataset_name)

    print("Tokenizing dataset")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Helper functions
    def remove_special_tokens(example):
        example['input'] = example['input'].replace('[CLS]', '')
        return {'text': example['input'], 'label': example['label']}

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    # Remove special tokens
    dataset = dataset.map(remove_special_tokens)
    # Tokenize
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    print("Loading model")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Move the model to the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    metric = evaluate.load("accuracy")

    print("Setting up training")
    # Set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = f"{model_name}-finetuned_{dataset_name.split('/ruletaker-')[1]}"
    # Save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "true"
    # Turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned_{dataset_name.split('/ruletaker-')[1]}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_grad_norm=1.0,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        num_train_epochs=args.num_train_epochs,
        push_to_hub=True,
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

if __name__ == "__main__":
    main()
