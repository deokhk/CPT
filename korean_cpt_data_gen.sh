#!/bin/bash

python crosslingual_pt_generator.py --preprocessed_text2sql_dataset /home/deokhk/project/Police_CPT/data/spider/preprocessed_train_spider.json \
--table_path /home/deokhk/project/Police_CPT/data/spider/tables.json \
--save_dir /home/deokhk/project/Police_CPT/data/spider/multitask_ft \
--db_path /mnt/hdd/deokhk/spider/database \
--translated_seq2seq_dataset_path /home/deokhk/project/Police_CPT/data/spider/train_spider_seq2seq_translated_ko.json