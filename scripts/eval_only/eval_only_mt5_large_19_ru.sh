set -e
# select the best text2sql-mt5-large ckpt
# 24GB -> 16 ok (might not ok in some cases..)
python -u evaluate_text2sql_ckpts.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/mt5-large-19-text2sql_ru" \
    --eval_results_path "./eval_results/mt5-large-19-text2sql_ru" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --cspider_dev_filepath "./data/preprocessed_data/dev_cspider_seq2seq.json" \
    --cspider_original_dev_filepath "./data/Cspider/dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --wandb_log \
    --exp_name "eval-mt5-large-19-ru" \
    --model_name_or_path "mt5-large"

python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/mt5-large-19-text2sql_ru/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_target_only_sql-ru" \
    --eval_file_name "eval_pauq.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_pauq_seq2seq.json" \
    --original_dev_filepath "./data/pauq/pauq_dev_converted.json" \
    --db_path "./pauq_db/merged_database" \
    --num_beams 8 \
    --num_return_sequences 8

# Second eval vanilla


python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large-target-only-sql (vanilla_do_not_erase)/best_model" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/text2sql-mt5-large_target_only_sql-vanilla" \
    --eval_file_name "eval_vanilla.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_pauq_seq2seq.json" \
    --original_dev_filepath "./data/pauq/pauq_dev_converted.json" \
    --db_path "./pauq_db/merged_database" \
    --num_beams 8 \
    --num_return_sequences 8

