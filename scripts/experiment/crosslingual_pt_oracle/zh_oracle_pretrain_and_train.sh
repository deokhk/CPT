#!/bin/bash
set -e

# train text2sql-mt0-base model
# Effective batch size: batch_size * gradient_accumulation_step * num_gpus
python -m torch.distributed.launch --nproc_per_node=4 crosslingual_pretraining.py \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2 \
    --num_training_steps 50000 \
    --num_eval_steps 2000 \
    --logging_steps 10 \
    --seed 42 \
    --save_path "./models/mt5-large-zh-cross-pt-oracle" \
    --model_name_or_path "google/mt5-large" \
    --sg_train_filepath "./data/crosslingual_pt_oracle_zh/sql_generation_train.json" \
    --sp_train_filepath "./data/crosslingual_pt_oracle_zh/schema_prediction_train.json" \
    --vp_train_filepath "./data/crosslingual_pt_oracle_zh/value_prediction_train.json" \
    --sg_val_filepath "./data/crosslingual_pt_oracle_zh/sql_generation_dev.json" \
    --sp_val_filepath "./data/crosslingual_pt_oracle_zh/schema_prediction_dev.json" \
    --vp_val_filepath "./data/crosslingual_pt_oracle_zh/value_prediction_dev.json" \
    --only_save_best_model \
    --wandb_log


# Stage-2 finetuning 

python -m torch.distributed.launch --nproc_per_node=4 text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed 42 \
    --save_path "./models/mt5-large-zh-cross-pt-oracle-text2sql" \
    --model_name_or_path "./models/mt5-large-zh-cross-pt-oracle/best_model" \
    --mode train \
    --train_filepath "./data/preprocessed_data/train_spider_seq2seq.json" \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq.json" \
    --wandb_log