#!/bin/bash

python -u evaluate_text2sql_ckpts.py \
    --batch_size 16 \
    --device "0" \
    --seed 42 \
    --save_path "/home1/deokhk_1/research/ZX-seq2seq/models/mt5-large-zh-cross-pt-encoder-gen-text2sql" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/crosslingual_zh_enc_gen" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --cross_dev_filepath "./data/preprocessed_data/dev_cspider_seq2seq.json" \
    --cross_original_dev_filepath "./data/Cspider/dev.json" \
    --cross_eval_dataset_name "cspider" \
    --db_path "./database" \
    --cross_db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --cross_eval_every_epoch \
    --wandb_log \
    --exp_name zh_encoder_gen_text2sql
