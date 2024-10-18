set -e
# select the best text2sql-mt5-base ckpt
# 24GB -> 16 ok (might not ok in some cases..)
python -u evaluate_text2sql_ckpts.py \
    --batch_size 8 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt5-base" \
    --model_name_or_path "google/mt5-base" \
    --eval_results_path "./eval_results/text2sql-mt5-base" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --cspider_dev_filepath "./data/preprocessed_data/dev_cspider_seq2seq.json" \
    --cspider_original_dev_filepath "./data/Cspider/dev.json" \
    --vspider_dev_filepath "./data/preprocessed_data/dev_vspider_seq2seq.json" \
    --vspider_original_dev_filepath "./data/Vspider/dev.json" \
    --db_path "./database" \
    --num_beams 8 \
    --num_return_sequences 8