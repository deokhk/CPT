#!/bin/bash

set -e
python translate_dataset.py --original_filepath "/home/deokhk/research/ZX-seq2seq/data/spider/train_spider.json" \
--target_lang "zh"

python translate_dataset.py --original_filepath "/home/deokhk/research/ZX-seq2seq/data/spider/train_spider.json" \
--target_lang "am"

python translate_dataset.py --original_filepath "/home/deokhk/research/ZX-seq2seq/data/spider/train_spider.json" \
--target_lang "vi"

python translate_dataset.py --original_filepath "/home/deokhk/research/ZX-seq2seq/data/spider/train_spider.json" \
--target_lang "ru"

python translate_dataset.py --original_filepath "/home/deokhk/research/ZX-seq2seq/data/spider/train_spider.json" \
--target_lang "te"
