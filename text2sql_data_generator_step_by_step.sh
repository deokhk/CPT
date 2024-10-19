#!/bin/bash

python text2sql_data_generator.py \
--input_dataset_path /home/deokhk/project/Police_CPT/data/spider/preprocessed_dev_spider.json \
--output_dataset_path /home/deokhk/project/Police_CPT/data/spider/dev_spider_seq2seq_stepgen.json \
--mode eval \
--use_contents \
--add_fk_info \
--stepgen