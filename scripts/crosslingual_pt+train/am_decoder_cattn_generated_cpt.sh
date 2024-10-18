#!/bin/bash
set -e

TRAINING_STEPS=20000
PRET_MODEL_SAVE_PATH=./models/mt5-large-am-cross-pt-decoder-cattn-gen
GENERATED_DATA_PATH=/home1/deokhk_1/research/XLang-NL2SQL/output/la_dec_cattn_only_mean_eng-mt5-sql2text-no-schema-50.16.3e-5.500/best_checkpoint/with_lang_generated_predictions_am_beam_4_from_train_spider_seq2seq_english.json
# Effective batch size: batch_size * gradient_accumulation_step * num_gpus
NUM_GPUS=4
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS crosslingual_pretraining.py \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --gradient_accumulation_steps 2 \
    --num_training_steps $TRAINING_STEPS \
    --num_eval_steps 1000 \
    --logging_steps 10 \
    --seed 42 \
    --save_path $PRET_MODEL_SAVE_PATH \
    --model_name_or_path "google/mt5-large" \
    --spider_train_filepath "/home1/deokhk_1/research/ZX-seq2seq/data/spider/train_spider.json" \
    --generated_question_sql_filepath $GENERATED_DATA_PATH \
    --table_path "./data/spider/tables.json" \
    --db_path "./database" \
    --use_contents \
    --add_fk_info \
    --only_save_best_model \
    --wandb_log 

# Stage-2 finetuning 

PRET_BEST_MODEL_PATH=./models/mt5-large-am-cross-pt-decoder-cattn-gen/best_model/
FT_MODEL_SAVE_PATH=./models/mt5-large-am-cross-pt-decoder-cattn-gen-text2sql
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed 42 \
    --save_path $FT_MODEL_SAVE_PATH \
    --model_name_or_path $PRET_BEST_MODEL_PATH \
    --mode train \
    --train_filepath "./data/preprocessed_data/train_spider_seq2seq.json" \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq.json" \
    --wandb_log