#!/bin/bash

set -e


python preprocessing.py \
    --mode "eval" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/amharic_spider/dev_amharic.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_amharic.json" \
    --db_path "./database"

python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_amharic.json" \
    --output_dataset_path "./data/preprocessed_data/dev_spider_seq2seq_amharic.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents