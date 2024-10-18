set -e

# train text2sql-mt5-large model
# Effective batch size: batch_size * gradient_accumulation_step * num_gpus
python -m torch.distributed.launch --nproc_per_node=2 text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large-target-only-sql" \
    --model_name_or_path "google/mt5-large" \
    --mode train \
    --train_filepath "./data/preprocessed_data/train_spider_seq2seq_target_only_sql.json" \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq_target_only_sql.json" \
    --wandb_log

# select the best text2sql-mt5-large ckpt
# 24GB -> 16 ok
python -u evaluate_text2sql_ckpts.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large-target-only-sql" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_target_only_sql" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq_target_only_sql.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --cspider_dev_filepath "./data/preprocessed_data/dev_cspider_seq2seq_target_only_sql.json" \
    --cspider_original_dev_filepath "./data/Cspider/dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8


# Evalaute on zh setting
python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large-target-only-sql/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_target_only_sql" \
    --eval_file_name "eval_zh.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_cspider_zh_seq2seq_target_only_sql.json" \
    --original_dev_filepath "./data/Cspider/zh_dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8

# Evaluate on translate-test setting (zh-full)
python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large-target-only-sql/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_target_only_sql" \
    --eval_file_name "eval_translate_test.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_cspider_seq2seq_translated_to_en_target_only_sql.json" \
    --original_dev_filepath "./data/Cspider/dev_translated_to_en.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8

# Evaluate on translate-test setting (zh)
python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large-target-only-sql/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_target_only_sql" \
    --eval_file_name "eval_zh_translate_test.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_cspider_zh_seq2seq_translated_to_en_target_only_sql.json" \
    --original_dev_filepath "./data/Cspider/zh_dev_translated_to_en.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8

