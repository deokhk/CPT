set -e

# train text2sql-mt5-large model
# baseline, no stepgen.
# Train date: 2023-11-07
# Trained on spider. For validation ckpt selection, we use EM+EXEC on Spider.
# Then, test on Cspider/Vspider/PAUQ/Amharic

# Effective batch size: batch_size * gradient_accumulation_step * num_gpus
python -m torch.distributed.launch --nproc_per_node=4 text2sql.py \
    --effective_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --epochs 50 \
    --seed 42 \
    --save_path "./models/text2sql-mt5-large_baseline" \
    --model_name_or_path "google/mt5-large" \
    --mode train \
    --train_filepath "./data/preprocessed_data/train_spider_seq2seq.json" \
    --dev_filepath "./data/preprocessed_data/dev_spider_seq2seq.json" \
    --wandb_log