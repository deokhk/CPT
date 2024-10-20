set -e

# preprocess Spider train dataset 
python preprocessing.py \
    --mode "train" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/spider/train_spider.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_train_spider.json" \
    --db_path "./database"

# preprocess dev dataset
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/spider/tables.json" \
    --input_dataset_path "./data/spider/dev.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_spider.json" \
    --db_path "./database"

# preprocess Cspider dev dataset 
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/Cspider/tables.json" \
    --input_dataset_path "./data/Cspider/dev.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider.json" \
    --db_path "./database"
    