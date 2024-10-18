#!/bin/bash 
set -e 

# First evaluate telugu 
python generate_and_evaluate_chatgpt_text2sql.py \
--openai_key_path "openai_key.txt" \
--model "gpt-3.5-turbo" \
--original_eval_dataset_path "./data/spider/dev_translated_to_te.json" \
--preprocessed_eval_dataset_path "./data/preprocessed_data/preprocessed_dev_spider_translated_to_te.json" \
--db_path "./database" \
--output_path "./predictions/spider_dev_translated_to_te_chatgpt_prediction.jsonl" \
--mode "eval" \
--use_contents \
--add_fk_info \
--reprocess


# Then evaluate amhara 
python generate_and_evaluate_chatgpt_text2sql.py \
--openai_key_path "openai_key.txt" \
--model "gpt-3.5-turbo" \
--original_eval_dataset_path "./data/spider/dev_translated_to_am.json" \
--preprocessed_eval_dataset_path "./data/preprocessed_data/preprocessed_dev_spider_translated_to_am.json" \
--db_path "./database" \
--output_path "./predictions/spider_dev_translated_to_am_chatgpt_prediction.jsonl" \
--mode "eval" \
--use_contents \
--add_fk_info \
--reprocess
