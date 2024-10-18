#!/bin/bash

set -e
python translate_dataset.py --original_dev_filepath "/home/deokhk/research/ZX-seq2seq/data/Cspider/zh_dev.json" \
--target_lang "en"

python preprocessing.py \
    --mode "eval" \
    --table_path "./data/Cspider/tables.json" \
    --input_dataset_path "./data/Cspider/zh_dev_translated_to_en.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider_zh_translated_to_en.json" \
    --db_path "./database"

python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider_zh_translated_to_en.json" \
    --output_dataset_path "./data/preprocessed_data/dev_cspider_zh_seq2seq_translated_to_en_target_only_sql.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents \
    --target_only_sql