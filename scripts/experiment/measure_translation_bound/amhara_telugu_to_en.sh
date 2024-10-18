#!/bin/bash

set -e

python translate_dataset.py --original_dev_filepath "./data/spider/dev_translated_to_am.json" --target_lang "en"

python preprocessing.py \
    --mode "eval" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/spider/dev_translated_to_am_translated_to_en.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_spider_translated_to_am_translated_to_en.json" \
    --db_path "./database"

python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_spider_translated_to_am_translated_to_en.json" \
    --output_dataset_path "./data/preprocessed_data/dev_spider_seq2seq_translated_to_am_translated_to_en_target_only_sql.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents \
    --target_only_sql


python translate_dataset.py --original_dev_filepath "./data/spider/dev_translated_to_te.json" --target_lang "en"

python preprocessing.py \
    --mode "eval" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/spider/dev_translated_to_te_translated_to_en.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_spider_translated_to_te_translated_to_en.json" \
    --db_path "./database"

python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_spider_translated_to_te_translated_to_en.json" \
    --output_dataset_path "./data/preprocessed_data/dev_spider_seq2seq_translated_to_te_translated_to_en_target_only_sql.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents \
    --target_only_sql
