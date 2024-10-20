set -e
# select the best text2sql-mt0-base ckpt
# 24GB -> 16 ok (might not ok in some cases..)
python -u evaluate_text2sql_ckpts.py \
    --batch_size 16 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt0-base" \
    --model_name_or_path "bigscience/mt0-base" \
    --eval_results_path "./eval_results/text2sql-mt0-base" \
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