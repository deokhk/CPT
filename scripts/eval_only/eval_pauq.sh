set -e
# select the best text2sql-mt5-large ckpt
# 24GB -> 16 ok (might not ok in some cases..)
python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/mt5-large-19-text2sql/checkpoint-43750" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/mt5-large-19-text2sql" \
    --eval_file_name "eval_zh.txt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_cspider_zh_seq2seq_target_only_sql.json" \
    --original_dev_filepath "./data/Cspider/zh_dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8

python -u evaluate_single_ckpt.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/mt5-large-19-text2sql (proposed)/checkpoint-43750/" \
    --model_name_or_path "google/mt5-large" \
    --eval_results_path "./eval_results/mono_am" \
    --eval_file_name "eval_mono_amtxt" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq_amharic.json" \
    --original_dev_filepath "./data/amharic_spider/dev_amharic.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8
