#!/bin/bash

python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large-target-only-sql (vanilla_do_not_erase)/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_target_only_sql" \
    --eval_file_name "eval_vspider.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_vspider_converted_seq2seq.json" \
    --original_dev_filepath "./data/Vspider/dev_converted.json" \
    --db_path "./vspider_database" \
    --num_beams 8 \
    --num_return_sequences 8