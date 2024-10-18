set -e

# generate text2sql training dataset
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_train_spider.json" \
    --output_dataset_path "./data/preprocessed_data/train_spider_seq2seq_target_only_sql.json" \
    --mode "train" \
    --add_fk_info \
    --use_contents \
    --target_only_sql

# generate text2sql development dataset
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_spider.json" \
    --output_dataset_path "./data/preprocessed_data/dev_spider_seq2seq_target_only_sql.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents \
    --target_only_sql

# generate text2sql Cspider development dataset
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider.json" \
    --output_dataset_path "./data/preprocessed_data/dev_cspider_seq2seq_target_only_sql.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents \
    --target_only_sql


# generate text2sql Cspider development dataset
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider.json" \
    --output_dataset_path "./data/preprocessed_data/dev_cspider_seq2seq_target_only_sql.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents \
    --target_only_sql

# generate text2sql Cspider development dataset (zh)
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider_zh.json" \
    --output_dataset_path "./data/preprocessed_data/dev_cspider_zh_seq2seq_target_only_sql.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents \
    --target_only_sql

