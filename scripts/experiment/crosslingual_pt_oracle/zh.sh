#!/bin/bash

python crosslingual_pt_generator.py \
    --generated_text2sql_data_path "./data/spider/train_spider_translated_to_zh.json" \
    --table_path "./data/spider/tables.json" \
    --save_dir "./data/crosslingual_pt_oracle_zh" \
    --db_path "./database" \
    --use_contents \
    --add_fk_info \
    --target_type "sql"
