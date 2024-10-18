#!/bin/bash 
set -e 

python generate_and_evaluate_chatgpt_text2sql.py \
--openai_key_path "openai_key.txt" \
--model "gpt-3.5-turbo" \
--original_eval_dataset_path "./data/pauq/pauq_dev_converted.json" \
--preprocessed_eval_dataset_path "./data/preprocessed_data/preprocessed_dev_pauq.json" \
--db_path "./database" \
--output_path "./predictions/pauq_dev_chatgpt_prediction.jsonl" \
--mode "eval" \
--use_contents \
--add_fk_info \
--reprocess