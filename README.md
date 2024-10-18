# Second stage: text2sql fine-tuning 

We support four pretrained multilingual models: mt5-base, mt5-large, mt0-base, mt0-large.

To reproduce the results, follow the steps below:

mT5-{base, large}
```
# Step1: preprocess spider dataset
sh deokhk_scripts/train/spider_text2sql/preprocess.sh
# Step2: prepare text-to-natsql training and development set for mT5
sh scripts/train/spider_text2sql/gen_seq2seq_dataset.sh
# Step3: fine-tune mT5-base 
sh scripts/train/spider_text2sql/train_mt5_base.sh
# Step3: (or) fine-tune mT5-large 
sh scripts/train/spider_text2sql/train_mt5_large.sh

```

mT0-{base, large}
```
# Step1: preprocess spider dataset
sh scripts/train/spider_text2sql/preprocess.sh
# Step2: prepare text-to-natsql training and development set for mT0
sh scripts/train/spider_text2sql/gen_seq2seq_dataset.sh
# Step3: fine-tune mT0-base 
sh scripts/train/spider_text2sql/train_mt0_base.sh
# Step3: (or) fine-tune mT0-large 
sh scripts/train/spider_text2sql/train_mt0_large.sh

```

# Other experiments
## Evaluating chatGPT zero-shot text2sql performance
First, you need to preprocess the spider dataset in each languages.
This can be done by executing following commands below.
```
sh scripts/train/spider_text2sql/preprocess.sh

```

Then, you can evaluate the zero-shot performance of chatGPT by executing following commands below.
```
sh scripts/experiment/cspider_chatgpt_inference.sh
```

Note that depending on the evaluation language, you should call different scripts. Above is an example for Chinese.



## Extracting Zh split from Cspider 
It is not appropriate to measure TS(test suite accuracy) across all examples in Cspider, as some queries contain chinese cell values.
Therefore, we extract the examples without chinese cell values and measure TS on them.
To extract such split(zh setting), execute the following commands below.

```
python cspider_zh_extrator.py --dev_path {path to cspider dev set} --output_dir {path to output dir}
```

Then, run the preprocessing scripts for the extracted split.

```
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/Cspider/tables.json" \
    --input_dataset_path "./data/Cspider/zh_dev.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider_zh.json" \
    --db_path "./database"
```

Now, convert to seq2seq format.

```
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider_zh.json" \
    --output_dataset_path "./data/preprocessed_data/dev_cspider_zh_seq2seq.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents
```
if you want stepgen-format data, then..
```
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_cspider_zh.json" \
    --output_dataset_path "./data/preprocessed_data/dev_cspider_zh_seq2seq_stepgen_sql.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents \
    --stepgen
```


## Test translate-test setting
To test such setting, we need to translate the test set to the target language(english).
Then, convert to preprocessed dataset and finally generate seq2seq dataset.
For Cspider, you can do so by simply executing the following commands below.
```
sh scripts/experiment/translate_test/cspider_to_en.sh
```


## PAUQ evaluation
You need to first download the dataset & database from https://github.com/ai-spiderweb/pauq
Move the downloaded files(dataset, tables.json) to `data/PAUQ` directory.
```
mkdir pauq_db
unzip merged_database_2022-06-10.zip -d ./pauq_db/
rm merged_database_2022-06-10.zip
```

First, you should convert dataset file.
```
python ./utils/pauq_converter.py --pauq_file_path ./data/pauq/pauq_dev.json

```

Preprocess the dataset
```
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/pauq/tables.json" \
    --input_dataset_path "./data/pauq/pauq_dev_converted.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_pauq.json" \
    --db_path "./pauq_db/merged_database" \
    --pauq
```
Convert to seq2seq format
```
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_pauq.json" \
    --output_dataset_path "./data/preprocessed_data/dev_pauq_seq2seq.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents \
```

When performing evaluation, make sure to set "pauq" flag to True.

## Make Vspider dataset executable 

First, copy and rename the database from Spider dataset.
```
cp -r database vspider_database
```

As SQLite does not support whitespace in either table name or column name, we need to rename the tables and columns.

To do so, execute the following commands below.
```
python ./utils/vspider_dataset_converter.py --table_path "./data/Vspider/tables.json" \
--dataset_path "./data/Vspider/dev.json"

```

Then, execute the following commands below to convert original spider databse to vspider database.
We replace column and table names in the original spider database with the ones in vspider database.
(From the root directory)
```
python ./utils/vspider_database_constructor.py --db_path "./vspider_database/" \
--spider_table_info "./data/spider/tables.json" \
--vspider_table_info "./data/Vspider/tables_converted.json"
```

Finally,
```
python preprocessing.py \
    --mode "eval" \
    --table_path "./data/Vspider/tables_converted.json" \
    --input_dataset_path "./data/Vspider/dev_converted.json" \
    --output_dataset_path "./data/preprocessed_data/preprocessed_dev_vspider_converted.json" \
    --db_path "./vspider_database"

python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_dev_vspider_converted.json" \
    --output_dataset_path "./data/preprocessed_data/dev_vspider_converted_seq2seq.json" \
    --mode "eval" \
    --add_fk_info \
    --use_contents

```

## Cross-lingual pretraining, followed by text2sql finetuning
First generate the cross lingual multitask pretraining dataset 
Go to ./scripts/experiment/crosslingual_pt_oracle
```
sh zh.sh

```
Then, train

## Proposed method: cross-lingual pretraining with synthetic data, followed by text2sql finetuning

