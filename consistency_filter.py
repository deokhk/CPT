import os 
import json 
import torch
import argparse 

from tqdm.auto import tqdm
from tokenizers import AddedToken

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MT5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers.trainer_utils import set_seed
from utils.load_dataset import Text2SQLDataset
from utils.text2sql_decoding_utils import decode_sqls
from third_party.spider.preprocess.get_tables import dump_db_json_schema
from utils.spider_metric.spider_test_suite import compute_test_suite_metric_sp


def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_synthetic_data_path", type=str, default="data", help="Path to synthesized data")
    parser.add_argument("--sql_synthetic_data_path", type=str, default="/home/deokhk/research/tensor2struct-public/experiments/sql2nl/data-synthetic/new-examples-no-question-6-128.json", help="Path to the data synthesized using PCFG")
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size. Note that this is a effective batch size')
    parser.add_argument('--device', type = str, default = "2",
                        help = 'the id of used GPU device.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')

    parser.add_argument("--model_name_or_path", type=str, default="t5-small", help="Path to pretrained NL2SQL model")
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')

    parser.add_argument('--db_path', type = str, default = "database",
                        help = 'file path of database.')

    opt = parser.parse_args()
    return opt


class FilterTool(object):
    def __init__(self):
        # self.args = args
        self.schema_cache = dict()
        self.golds = []

    def register_golds(self, dataset, db_path):
        for idx, sample in enumerate(dataset):
            # To match the format..

            if sample['query'] == 'SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1':
                sample['query'] = 'SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1'
                sample['query_toks'] = ['SELECT', 'T1.company_type', 'FROM', 'Third_Party_Companies', 'AS', 'T1', 'JOIN', 'Maintenance_Contracts', 'AS', 'T2', 'ON', 'T1.company_id', '=', 'T2.maintenance_contract_company_id', 'ORDER', 'BY', 'T2.contract_end_date', 'DESC', 'LIMIT', '1']
                sample['query_toks_no_value'] =  ['select', 't1', '.', 'company_type', 'from', 'third_party_companies', 'as', 't1', 'join', 'maintenance_contracts', 'as', 't2', 'on', 't1', '.', 'company_id', '=', 't2', '.', 'maintenance_contract_company_id', 'order', 'by', 't2', '.', 'contract_end_date', 'desc', 'limit', 'value']
                sample['question'] = 'What is the type of the company who concluded its contracts most recently?'
                sample['question_toks'] = ['What', 'is', 'the', 'type', 'of', 'the', 'company', 'who', 'concluded', 'its', 'contracts', 'most', 'recently', '?']
            if sample['query'].startswith('SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN'):
                sample['query'] = sample['query'].replace('IN (SELECT T2.dormid)', 'IN (SELECT T3.dormid)')
                index = sample['query_toks'].index('(') + 2
                assert sample['query_toks'][index] == 'T2.dormid'
                sample['query_toks'][index] = 'T3.dormid'
                index = sample['query_toks_no_value'].index('(') + 2
                assert sample['query_toks_no_value'][index] == 't2'
                sample['query_toks_no_value'][index] = 't3'

            db_id = sample["db_id"]
            if db_id not in self.schema_cache:
                self.schema_cache[db_id] = dump_db_json_schema(
                    db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                )
            schema = self.schema_cache[db_id]

            self.golds.append({
                "query": sample["query"],
                "question": sample["question"],
                "db_id": db_id,
                "db_path": db_path,
                "db_table_names": schema["table_names_original"],
                "db_column_names": {
                    "table_id": [table_id for table_id, _ in schema["column_names_original"]],
                    "column_name": [column_name for _, column_name in schema["column_names_original"]]
                },
                "db_column_types": schema["column_types"],
                "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                "db_foreign_keys": {
                    "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
                    "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]]
                },
            })

    
    def filter_dataset(self, preds):
        eval_res = compute_test_suite_metric_sp(preds, self.golds, db_dir = None)
        return {**eval_res}




def predict_SQL(opt):
        
    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, AutoTokenizer):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    generated_dataset = Text2SQLDataset(
        dir_ = opt.question_synthetic_data_path,
        mode = "eval"
    )

    gen_dataloader = DataLoader(
        generated_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else AutoModelForSeq2SeqLM

    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    # initialize model
    model = model_class.from_pretrained(opt.model_name_or_path)
    if torch.cuda.is_available():
        model = model.to(device)

    model.eval()
    predict_sqls = []
    for batch in tqdm(gen_dataloader):
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
    
    return predict_sqls

def filter_dataset(opt):
    # Note : for test, we didn't apply acclerators due to complexity of inference
    set_seed(opt.seed)

    predict_sqls = []
    predict_sqls = predict_SQL(opt)
    
    
    # initialize evaluator
    filter = FilterTool()

    # Construct gold dataset 
    with open(opt.sql_synthetic_data_path, encoding="utf-8") as f:
        sql_syn_dataset = json.load(f)
    with open(opt.question_synthetic_data_path, encoding="utf-8") as f:
        question_syn_dataset = json.load(f)
    
    gold_dataset = []

    if len(sql_syn_dataset) > len(question_syn_dataset):
        sql_syn_dataset = sql_syn_dataset[:len(question_syn_dataset)]

    for (sql_syn, q_syn) in zip(sql_syn_dataset, question_syn_dataset):
        assert sql_syn["db_id"] == q_syn["db_id"]
        gold_dataset.append({
            "db_id": sql_syn["db_id"],
            "query": sql_syn["query"],
            "question": q_syn["generated_question"]
        })
    
    filter.register_golds(gold_dataset, opt.db_path)
    res = filter.filter_dataset(predict_sqls)
    exec_score = res["exec"]
    eval_res = res["eval_res"]
    eval_error_rate = res["eval_error_rate"]
    print(f"exec_score : {exec_score}")
    print(f"eval_error_rate : {eval_error_rate}")

    filtered_question_syn_dataset = [x for x, exec_ in zip(question_syn_dataset, eval_res) if exec_ == 1]

    print(f"Filtered dataset size : {len(filtered_question_syn_dataset)}")

    return filtered_question_syn_dataset


if __name__ == "__main__":
    opt = parse_option()    
    filtered_dataset = filter_dataset(opt)

    save_dir = os.path.dirname(opt.question_synthetic_data_path)
    save_filename = "filtered_" + os.path.basename(opt.question_synthetic_data_path)
    save_path = os.path.join(save_dir, save_filename)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(filtered_dataset, f, indent=4, ensure_ascii=False)
    print(f"Filtered dataset saved in {save_path}")
    # Save 

