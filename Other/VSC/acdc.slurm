#!/bin/bash

#SBATCH --job-name=acdc_bert_text-entailment
#SBATCH --output=acdc_bert_text-entailment.out
#SBATCH --error=acdc_bert_text-entailment.err

#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=9
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=3400

cd $VSC_DATA/acdc/Automatic-Circuit-Discovery/acdc
source activate acdc

python main_custom.py --task=text-entailment --nexamples=20 --threshold=0.02 --using-wandb --model="andres-vs/bert-base-uncased-finetuned_Att-Noneg-depth0"

