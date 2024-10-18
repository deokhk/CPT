#!/bin/bash

set -e

# train mt5-large model
# Effective batch size: batch_size * gradient_accumulation_step * num_gpus
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes 1 --rdzv_backend c10d --rdzv_endpoint localhost:0 crosslingual_pretraining.py \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2 \
    --num_training_steps 50000 \
    --num_eval_steps 2000 \
    --logging_steps 50 \
    --seed 42 \
    --save_path "/mnt/hdd/deokhk/police_korean_cpt" \
    --model_name_or_path "google/mt5-large" \
    --sg_train_filepath "./data/spider/multitask_ft/sql_generation_train.json" \
    --sp_train_filepath "./data/spider/multitask_ft/schema_prediction_train.json" \
    --vp_train_filepath "./data/spider/multitask_ft/value_prediction_train.json" \
    --sg_val_filepath "./data/spider/multitask_ft/sql_generation_dev.json" \
    --sp_val_filepath "./data/spider/multitask_ft/schema_prediction_dev.json" \
    --vp_val_filepath "./data/spider/multitask_ft/value_prediction_dev.json" \
    --wandb_log \
    --only_save_best_model
