import os
import json
import torch
import argparse
import torch.optim as optim
import transformers
import wandb 
import torch.nn as nn

from tqdm.auto import tqdm
from tokenizers import AddedToken
from accelerate import Accelerator

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from utils.spider_metric.evaluator import EvaluateTool
from utils.load_dataset import Text2SQLDataset
from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--effective_batch_size', type = int, default = 8,
                        help = 'input batch size. Should be effective batch size!')
    parser.add_argument('--gradient_accumulation_steps', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_accumulation_step" steps.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--epochs', type = int, default = 50,
                        help = 'training epochs.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "models/text2sql",
                        help = 'save path of best fine-tuned text2sql model.')
    parser.add_argument('--wandb_log', action="store_true", help="Enable for wandb logging")
    parser.add_argument('--model_name_or_path', type = str, default = "t5-3b",
                        help = 
                        '''
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')
    parser.add_argument('--use_adafactor', action='store_true',
                        help = 'whether to use adafactor optimizer.')
    parser.add_argument('--mode', type = str, default = "train",
                        help='train, eval or test.')
    parser.add_argument('--train_filepath', type = str, default = "data/preprocessed_data/resdsql_train_spider.json",
                        help = 'file path of test2sql training set.')
    parser.add_argument('--dev_filepath', type = str, default = "data/preprocessed_data/resdsql_dev.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--original_dev_filepath', type = str, default = "data/spider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "database",
                        help = 'file path of database.')
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--output", type = str, default = "predicted_sql.txt",
                help = "save file of the predicted sqls.")
    parser.add_argument("--local_rank", type=int)
    opt = parser.parse_args()

    return opt

def _train(opt):
    set_seed(opt.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
    )

    is_local_main_process = accelerator.is_local_main_process

    if is_local_main_process:
        print(opt)

    if opt.wandb_log and is_local_main_process:
        wandb.init(
            project="ZX_seq2seq",
            name=f"{opt.model_name_or_path}",
        )


    text2sql_tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )

    if isinstance(text2sql_tokenizer, AutoTokenizer):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])


    # Compute batch size per gpu, by considering number of gpus and gradient accumulation steps.
    opt.batch_size = opt.effective_batch_size // opt.gradient_accumulation_steps // torch.cuda.device_count()

    assert opt.effective_batch_size == opt.batch_size * opt.gradient_accumulation_steps * torch.cuda.device_count()
    
    if is_local_main_process:
        print(f"batch size per gpu: {opt.batch_size}")
        print(f"gradient accumulation steps: {opt.gradient_accumulation_steps}")
        print(f"number of gpus: {torch.cuda.device_count()}")
        print(f"effective batch size: {opt.effective_batch_size}")


    train_dataset = Text2SQLDataset(
        dir_ = opt.train_filepath,
        mode = "train"
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size = opt.batch_size, 
        shuffle = True,
        collate_fn = lambda x: x,
        drop_last = True
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    if is_local_main_process:
        print("initializing text2sql model.")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path)
    model.resize_token_embeddings(len(text2sql_tokenizer))

    device = accelerator.device

    # TODO: remove this!
    if is_local_main_process:
        print(f"train_dataset length: {len(train_dataset)}")
    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.batch_size)
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.batch_size)

    if opt.use_adafactor:
        if is_local_main_process:
            print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        if is_local_main_process:
            print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model_decoder_start_token_id = model.config.decoder_start_token_id

    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    train_step = 0

    for epoch in range(opt.epochs):
        # Training 
        if is_local_main_process:
            print(f"This is epoch {epoch+1}.")
        model.train()

        train_pbar = tqdm(train_dataloader, disable=not is_local_main_process, desc="Training..")
        for batch in train_pbar:
            with accelerator.accumulate(model):
                train_step += 1
                
                batch_inputs = [data[0] for data in batch]
                batch_sqls = [data[1] for data in batch]
                batch_db_ids = [data[2] for data in batch] # unused
                batch_tc_original = [data[3] for data in batch] # unused
                
                tokenized_inputs = text2sql_tokenizer(
                    batch_inputs, 
                    padding = "max_length",
                    return_tensors = "pt",
                    max_length = 512,
                    truncation = True
                )
                
                with text2sql_tokenizer.as_target_tokenizer():
                    tokenized_outputs = text2sql_tokenizer(
                        batch_sqls, 
                        padding = "max_length", 
                        return_tensors = 'pt',
                        max_length = 256,
                        truncation = True
                    )
                
                encoder_input_ids = tokenized_inputs["input_ids"]
                encoder_input_attention_mask = tokenized_inputs["attention_mask"]

                decoder_labels = tokenized_outputs["input_ids"]
                decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
                decoder_attention_mask = tokenized_outputs["attention_mask"]

                if torch.cuda.is_available():
                    encoder_input_ids = encoder_input_ids.to(device)
                    encoder_input_attention_mask = encoder_input_attention_mask.to(device)
                    decoder_labels = decoder_labels.to(device)
                    decoder_attention_mask = decoder_attention_mask.to(device)
                
                model_outputs = model(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    labels = decoder_labels,
                    decoder_attention_mask = decoder_attention_mask,
                    return_dict = True
                )
                
                loss = model_outputs["loss"]
                accelerator.backward(loss)
                if opt.wandb_log and is_local_main_process:
                    wandb.log({"train loss": loss.item(), "train lr": optimizer.state_dict()['param_groups'][0]['lr']}, step=train_step)
                elif not opt.wandb_log and is_local_main_process:
                    print(f"At {train_step} training step, loss = {loss.mean().item()}.")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if is_local_main_process:
            print(f"At {train_step} training step, save a checkpoint.")
            os.makedirs(opt.save_path, exist_ok = True)

        accelerator.wait_for_everyone()
        
        unwrapped_model = accelerator.unwrap_model(model)
        if is_local_main_process and epoch >= 5:
            # Without pretraining, We only save model from epoch 5, since model with earlier epoch produced invalid sql, resulting in massive evaluation time
            save_directory = opt.save_path + "/checkpoint-{}".format(train_step)
            unwrapped_model.save_pretrained(save_directory = save_directory)
            print(f"Checkpoint saved at {save_directory}.")
            text2sql_tokenizer.save_pretrained(save_directory = save_directory)

    wandb.finish()

def _predict(opt):

    import time
    start_time = time.time()
        
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        opt.save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, AutoTokenizer):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    dev_dataset = Text2SQLDataset(
        dir_ = opt.dev_filepath,
        mode = opt.mode
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    # initialize model
    model = model_class.from_pretrained(opt.save_path)
    if torch.cuda.is_available():
        model = model.to(device)

    model.eval()
    predict_sqls = []
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.to(device)
            encoder_input_attention_mask = encoder_input_attention_mask.to(device)

        with torch.no_grad():
            model_outputs = model.generate(
                input_ids = encoder_input_ids,
                attention_mask = encoder_input_attention_mask,
                max_length = 256,
                decoder_start_token_id = model.config.decoder_start_token_id,
                num_beams = opt.num_beams,
                num_return_sequences = opt.num_return_sequences
            )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
            predict_sqls += decode_sqls(
                opt.db_path, 
                model_outputs, 
                batch_db_ids, 
                batch_inputs, 
                tokenizer, 
                batch_tc_original
            )
    
    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)
    
    # save results
    with open(opt.output, "w", encoding = 'utf-8') as f:
        for pred in predict_sqls:
            f.write(pred + "\n")
    
    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))
    return predict_sqls


def _test(opt):
    # Note : for test, we didn't apply acclerators due to complexity of inference
    set_seed(opt.seed)
    print(opt)

    predict_sqls = []
    """
    if os.path.exists(opt.output):
        print("Output file already exists. Please delete it if you want to re-generate it.")
        # read output file line by line and store it in predict_sqls
        with open(opt.output, "r", encoding = 'utf-8') as f:
            for line in f:
                predict_sqls.append(line.strip())
    """
    predict_sqls = _predict(opt)
    
    
    if opt.mode == "eval":
        # initialize evaluator
        evaluator = EvaluateTool()
        evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
        spider_metric_result = evaluator.evaluate(predict_sqls)
        print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
        print('exec score: {}'.format(spider_metric_result["exec"]))
    
        return spider_metric_result["exact_match"], spider_metric_result["exec"]


if __name__ == "__main__":
    opt = parse_option()
    if opt.mode in ["train"]:
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        _test(opt)