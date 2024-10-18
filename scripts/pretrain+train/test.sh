#!/bin/bash
set -e

python -m torch.distributed.launch --nproc_per_node=2 multitask_finetuning.py \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2 \
    --num_training_steps 100000 \
    --num_eval_steps 2000 \
    --logging_steps 50 \
    --seed 42 \
    --save_path "./models/mt0-large_multi_ft_16" \
    --model_name_or_path "bigscience/mt0-large" \
    --mt_train_filepath "./data/multitask_ft/translation_train.json" \
    --sp_train_filepath "./data/multitask_ft/schema_prediction_train.json" \
    --vp_train_filepath "./data/multitask_ft/value_prediction_train.json" \
    --mt_val_filepath "./data/multitask_ft/translation_val.json" \
    --sp_val_filepath "./data/multitask_ft/schema_prediction_val.json" \
    --vp_val_filepath "./data/multitask_ft/value_prediction_val.json" \
    --k 65536 \
    --only_save_best_model
