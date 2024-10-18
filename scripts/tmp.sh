#!/bin/bash

set -e

# stage-2 finetuning

python -m torch.distributed.launch --nproc_per_node=4 text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed 42 \
    --save_path "./models/mt0-large-19-text2sql" \
    --model_name_or_path "./models/mt0-large_multi_ft_19/best_model" \
    --mode train \
    --train_filepath "./data/preprocessed_data/train_spider_seq2seq.json" \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq.json" \
    --wandb_log
