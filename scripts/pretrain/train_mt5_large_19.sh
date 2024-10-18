#!/bin/bash
set -e

# train mt5-large model
# Effective batch size: batch_size * gradient_accumulation_step * num_gpus
torchrun --standalone --nnodes=1 --nproc_per_node= multitask_finetuning.py \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2 \
    --num_training_steps 100000 \
    --num_eval_steps 2000 \
    --logging_steps 50 \
    --seed 42 \
    --save_path "./models/mt5-large_multi_ft_19" \
    --model_name_or_path "google/mt5-large" \
    --mt_train_filepath "./data/multitask_ft/translation_train.json" \
    --sp_train_filepath "./data/multitask_ft/schema_prediction_train.json" \
    --vp_train_filepath "./data/multitask_ft/value_prediction_train.json" \
    --mt_val_filepath "./data/multitask_ft/translation_val.json" \
    --sp_val_filepath "./data/multitask_ft/schema_prediction_val.json" \
    --vp_val_filepath "./data/multitask_ft/value_prediction_val.json" \
    --k 524288 \
    --wandb_log \
    --only_save_best_model