set -e

# train text2sql-mt0-large model
# Effective batch size: batch_size * gradient_accumulation_step * num_gpus
python -m torch.distributed.launch --nproc_per_node=4 text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed 42 \
    --save_path "./models/text2sql-mt0-large" \
    --model_name_or_path "bigscience/mt0-large" \
    --mode train \
    --train_filepath "./data/preprocessed_data/train_spider_seq2seq.json" \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq.json" \
    --wandb_log

# select the best text2sql-mt0-large ckpt# 24GB -> 16 ok
python -u evaluate_text2sql_ckpts.py \
    --batch_size 16 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2sql-mt0-large" \
    --model_name_or_path "bigscience/mt0-large" \
    --eval_results_path "./eval_results/text2sql-mt0-large" \
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
