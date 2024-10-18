#!/bin/bash
set -e

# train text2sql-mt0-base model
# Effective batch size: batch_size * gradient_accumulation_step * num_gpus
python -m torch.distributed.launch --nproc_per_node=4 multitask_finetuning.py \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 1 \
    --num_training_steps 100000 \
    --num_eval_steps 200 \
    --logging_steps 10 \
    --seed 42 \
    --save_path "./models/mt0-base_multi_ft" \
    --model_name_or_path "bigscience/mt0-base" \
    --mt_train_filepath "./data/multitask_ft/translation_train.json" \
    --sp_train_filepath "./data/multitask_ft/schema_prediction_train.json" \
    --vp_train_filepath "./data/multitask_ft/value_prediction_train.json" \
    --mt_val_filepath "./data/multitask_ft/translation_val_sample.json" \
    --sp_val_filepath "./data/multitask_ft/schema_prediction_val_sample.json" \
    --vp_val_filepath "./data/multitask_ft/value_prediction_val_sample.json" \
    --k 65536 \
    --wandb_log 