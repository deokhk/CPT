#!/bin/bash

DATA_PATH=/home/deokhk/research/ZX-seq2seq/data/RT_consistency/OneM_wschema_generated_predictions_zh_beam_4_from_spider_new-examples-no-question-6-128_seq2seq_sampled.json
ORIGINAL_DATA_PATH=/home/deokhk/research/tensor2struct-public/experiments/sql2nl/data-synthetic/new-examples-no-question-6-128.json
MODEL_PATH=/home/deokhk/research/XLang-NL2SQL/output/text2sql-mt5-large_baseline/best_model
CUDA_VISIBLE_DEVICES=0 python consistency_filter.py \
    --question_synthetic_data_path $DATA_PATH \
    --sql_synthetic_data_path $ORIGINAL_DATA_PATH \
    --batch_size 4 \
    --device "0" \
    --model_name_or_path $MODEL_PATH \
    --db_path "./database"