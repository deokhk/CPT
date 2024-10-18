import json
import copy
import argparse
import random
import numpy as np
from tqdm import tqdm 

def parse_option():
    parser = argparse.ArgumentParser("command line arguments for generating the ranked dataset.")
    
    parser.add_argument('--input_dataset_path', type = str, default = "./data/pre-processing/dev_with_probs.json",
                        help = 'filepath of the input dataset.')
    parser.add_argument('--output_dataset_path', type = str, default = "./data/pre-processing/resdsql_dev.json",
                        help = 'filepath of the output dataset.')
    parser.add_argument('--mode', type = str, default = "eval",
                        help = 'type of the input dataset, options: train, eval, test.')
    parser.add_argument('--use_contents', action = 'store_true',
                        help = 'whether to add database contents in the input sequence.')
    parser.add_argument('--add_fk_info', action = 'store_true',
                    help = 'whether to add foreign key in the input sequence.')
    parser.add_argument("--target_type", type = str, default = "sql",
                help = "sql or natsql.")
    parser.add_argument("--stepgen", action = "store_true",
                help = "Apply step-by-step generation for sql.")

    opt = parser.parse_args()

    return opt

def lista_contains_listb(lista, listb):
    for b in listb:
        if b not in lista:
            return 0
    
    return 1

def prepare_input_and_output(opt, preprocessed_data):
    # Input-output format
    # Input: Translate the following sequence into SQL:{question} | {schema sequence}
    # Output: <sql> select count(singer_id) from singer

    # If stepgen is on,
    # Input : Translate the following sequence into SQL, after listing schema items and values mentioned in the utterance:{question} | {schema sequence}
    # Output: <table> singer <column> singer_id <value> <none> <sql> select count(singer_id) from singer

    question = preprocessed_data["question"]

    schema_sequence = ""
    for table_id in range(len(preprocessed_data["db_schema"])):
        table_name_original = preprocessed_data["db_schema"][table_id]["table_name_original"]
        # add table name
        schema_sequence += " | " + table_name_original + " : "
        
        column_info_list = []
        for column_id in range(len(preprocessed_data["db_schema"][table_id]["column_names_original"])):
            # extract column name
            column_name_original = preprocessed_data["db_schema"][table_id]["column_names_original"][column_id]
            db_contents = preprocessed_data["db_schema"][table_id]["db_contents"][column_id]
             # use database contents if opt.use_contents = True
            if opt.use_contents and len(db_contents) != 0:
                column_contents = " , ".join(db_contents)
                column_info = table_name_original + "." + column_name_original + " ( " + column_contents + " ) "
            else:
                column_info = table_name_original + "." + column_name_original

            column_info_list.append(column_info)
            
        # add column names
        schema_sequence += " , ".join(column_info_list)

    if opt.add_fk_info:
        for fk in preprocessed_data["fk"]:
            schema_sequence += " | " + fk["source_table_name_original"] + "." + fk["source_column_name_original"] + \
                " = " + fk["target_table_name_original"] + "." + fk["target_column_name_original"]
    
    # remove additional spaces in the schema sequence
    while "  " in schema_sequence:
        schema_sequence = schema_sequence.replace("  ", " ")


    # input_sequence = input_prefix + question + schema sequence
    if opt.stepgen:
        input_prefix = "Translate the following sequence into SQL, after listing schema items and values mentioned in the utterance:"
    else:
        input_prefix = "Translate the following sequence into SQL:"
    input_sequence = input_prefix + question + schema_sequence

    # Find mentioned schema items and values
    mentioned_table_ids = [idx for idx, label in enumerate(preprocessed_data["table_labels"]) if label == 1]
    mentioned_table_original_names = []
    mentioned_column_original_names = []

    for table_id in mentioned_table_ids:
        mentioned_table_original_names.append(preprocessed_data["db_schema"][table_id]["table_name_original"])
        mentioned_column_ids = [idx for idx, column_label in enumerate(preprocessed_data["column_labels"][table_id]) if column_label == 1]

        mentioned_column_original_names += [preprocessed_data["db_schema"][table_id]["column_names_original"][column_id] for column_id in mentioned_column_ids]

    o_t = ""
    if len(mentioned_table_original_names) != 0:
        o_t = ", ".join(mentioned_table_original_names)
    else:
        o_t = "<none>"
    
    o_c = ""
    if len(mentioned_column_original_names) != 0:
        o_c = ", ".join(mentioned_column_original_names)
    else:   
        o_c = "<none>"

    # Find mentioned values 
    mentioned_values = preprocessed_data["sql_values"]
    o_v = ""
    if len(mentioned_values) != 0:
        o_v = ", ".join(mentioned_values)
    else:
        o_v = "<none>"

    if opt.stepgen:
        output_sequence = "<table>" +  " " + o_t + " " + "<column>" + " " + o_c + " " + "<value>" + " " + o_v + " " + "<sql>" + " " + preprocessed_data["norm_sql"]
    else:
        output_sequence = "<sql>" + " " + preprocessed_data["norm_sql"]
    
    return input_sequence, output_sequence

def generate_seq2seq_dataset(opt):
    with open(opt.input_dataset_path) as f:
        dataset = json.load(f)
    
    output_dataset = []
    for data_id, data in tqdm(enumerate(dataset), desc="Generating seq2seq dataset"):
        preprocessed_data = dict()
        preprocessed_data["question"] = data["question"]
        preprocessed_data["sql"] = data["sql"] # unused
        preprocessed_data["norm_sql"] = data["norm_sql"]
        preprocessed_data["db_id"] = data["db_id"]
        preprocessed_data["db_schema"] = []

        # record table & column labels
        preprocessed_data["table_labels"] = data["table_labels"]
        preprocessed_data["column_labels"] = data["column_labels"]
        preprocessed_data["sql_values"] = data["sql_values"]

        table_ids = [idx for idx, _ in enumerate(data["db_schema"])]
        
        for table_id in table_ids:
            new_table_info = dict()
            new_table_info["table_name_original"] = data["db_schema"][table_id]["table_name_original"]
            # record ids of used columns

            new_table_info["column_names_original"] = data["db_schema"][table_id]["column_names_original"]
            new_table_info["db_contents"] = data["db_schema"][table_id]["db_contents"]
            
            preprocessed_data["db_schema"].append(new_table_info)

        # record foreign keys
        preprocessed_data["fk"] = data["fk"]

        input_sequence, output_sequence = prepare_input_and_output(opt, preprocessed_data)
        
        # record table_name_original.column_name_original for subsequent correction function during inference
        tc_original = []
        for table in preprocessed_data["db_schema"]:
            for column_name_original in ["*"] + table["column_names_original"]:
                tc_original.append(table["table_name_original"] + "." + column_name_original)

        output_dataset.append(
            {
                "db_id": data["db_id"],
                "input_sequence": input_sequence, 
                "output_sequence": output_sequence,
                "tc_original": tc_original
            }
        )
    
    with open(opt.output_dataset_path, "w") as f:
        f.write(json.dumps(output_dataset, indent = 2, ensure_ascii = False))

if __name__ == "__main__":
    opt = parse_option()
    random.seed(42)
    
    generate_seq2seq_dataset(opt)
