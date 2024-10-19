#!/bin/bash


set -e

# train text2sql-mt5-large model
# Effective batch size: batch_size * gradient_accumulation_step * num_gpus
torchrun --nproc_per_node=4 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed 42 \
    --save_path "/home1/deokhk_1/project/CPT/text2sql_models/" \
    --model_name_or_path "/home1/deokhk_1/project/CPT/cpt_models/best_model/" \
    --mode train \
    --train_filepath "/home1/deokhk_1/project/CPT/data/spider/train_spider_seq2seq_stepgen.json" \
    --dev_filepath "/home1/deokhk_1/project/CPT/data/spider/dev_spider_seq2seq_stepgen.json" \
    --wandb_log

# select the best text2sql-mt5-large ckpt
# 24GB -> 16 ok
CUDA_VISIBLE_DEVICES=0 python -u evaluate_text2sql_ckpts.py \
    --batch_size 16 \
    --device "0" \
    --seed 42 \
    --save_path "/home1/deokhk_1/project/CPT/text2sql_models/" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large" \
    --mode eval \
    --dev_filepath "./data/spider/dev_spider_seq2seq_stepgen.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path ""/home1/deokhk_1/research/ZX-seq2seq/database/"" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --wandb_log