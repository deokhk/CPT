#!/bin/bash 
set -e 

python generate_and_evaluate_chatgpt_text2sql.py \
--openai_key_path "openai_key.txt" \
--model "gpt-3.5-turbo" \
--original_eval_dataset_path "./data/Vspider/dev_converted.json" \
--preprocessed_eval_dataset_path "./data/preprocessed_data/preprocessed_dev_vspider_converted.json" \
--db_path "./vspider_database" \
--output_path "./predictions/vspider_dev_chatgpt_prediction.jsonl" \
--mode "eval" \
--use_contents \
--add_fk_info \
--reprocess