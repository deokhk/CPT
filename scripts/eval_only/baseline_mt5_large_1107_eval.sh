
# 24GB -> 4 ok
python -u evaluate_text2sql_ckpts.py \
    --batch_size 4 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large_baseline" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --cross_dev_filepath "./data/preprocessed_data/dev_cspider_seq2seq.json" \
    --cross_original_dev_filepath "./data/Cspider/dev.json" \
    --cross_eval_dataset_name "cspider" \
    --db_path "./database" \
    --cross_db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8


# On Vspider

python -u evaluate_single_ckpt.py \
    --batch_size 4 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large_baseline/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_baseline-vspider" \
    --eval_file_name "eval_vspider.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_vspider_converted_seq2seq.json" \
    --original_dev_filepath "./data/Vspider/dev_converted.json" \
    --db_path "./vspider_database" \
    --num_beams 8 \
    --num_return_sequences 8


# On PAUQ

python -u evaluate_single_ckpt.py \
    --batch_size 4 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large_baseline/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_baseline-pauq" \
    --eval_file_name "eval_pauq.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_pauq_seq2seq.json" \
    --original_dev_filepath "./data/pauq/pauq_dev_converted.json" \
    --db_path "./pauq_db/merged_database" \
    --num_beams 8 \
    --num_return_sequences 8

# On Amharic 

python -u evaluate_single_ckpt.py \
    --batch_size 4 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large_baseline/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_baseline-ahmaric_spider" \
    --eval_file_name "eval_amharic_spider.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq_amharic.json" \
    --original_dev_filepath "./data/amharic_spider/dev_amharic.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8

# On zh
python -u evaluate_single_ckpt.py \
    --batch_size 4 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large_baseline/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_baseline-cspider-zh" \
    --eval_file_name "eval_zh_spider.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_cspider_zh_seq2seq.json" \
    --original_dev_filepath "./data/Cspider/zh_dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8


# On zh-full
python -u evaluate_single_ckpt.py \
    --batch_size 4 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large_baseline/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_baseline-cspider" \
    --eval_file_name "eval_cspider.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_cspider_seq2seq.json" \
    --original_dev_filepath "./data/Cspider/dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8
