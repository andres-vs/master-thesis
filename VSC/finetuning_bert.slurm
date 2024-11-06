#!/bin/bash

#SBATCH --job-name=finetuning_bert
#SBATCH --output=finetuning_bert.out
#SBATCH --error=finetuning_bert.err

#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=3400

cd $VSC_SCRATCH/finetuning_bert
source activate finetuning_bert

python $VSC_DATA/finetuning_bert/finetuning_BERT_VSC_with_arguments.py --model_name=bert-base-uncased --dataset_name=andres-vs/ruletaker-Att-Noneg-depth1 --batch_size=16 --learning_rate=1e-5 --num_train_epochs=20
