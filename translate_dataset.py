# This file will translate given input file to translated version (using google API)
# dev.json and dev_seq2seq.json

# For evaluating translate-test setting!

# Except Vspider, all other dataset will only translate utterance into english
# As schema is in english, and we want to keep the schema as it is.

# Note that typically, the sum of chracter of both files are less than 50K. (for train split, it is roughly 500K)

import argparse 
import json 
import os 
import logging 
import six
from google.cloud import translate_v2 as translate
from nltk.tokenize import word_tokenize

from tqdm import tqdm 


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/deokhk/research/ZX-seq2seq/translator-itrc-cdbcc3fd38d7.json"
translate_client = translate.Client()


def translate_text(translate_client, text, target="en"):
    """
    Given a text(string), translate it into a target language. (english by default)
    return translated version of the given text.
    """    
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    result = translate_client.translate(text, target_language=target)

    return result["translatedText"]



def main(args):
    
    with open(args.original_filepath, "r") as f:
        original_data = json.load(f)
    
    translated_original_data = []

    
    # For original dev data, we need to translate question only.
    for original in tqdm(original_data, desc=f"Translating original dataset to {args.target_lang}"):
        orginal_question = original["question"]
        translated_question = translate_text(translate_client, orginal_question, target=args.target_lang)
        translated_question_toks = word_tokenize(translated_question)
        translated_original_data.append(
            {
                "db_id": original["db_id"],
                "query_toks_no_value": original["query_toks_no_value"],
                "question_toks": translated_question_toks,
                "query_toks": original["query_toks"],
                "question": translated_question,
                "original_question": orginal_question,
                "sql": original["sql"],
                "query": original["query"]
            }
        )


    original_file_dir = os.path.dirname(args.original_filepath)
    original_filename = os.path.basename(args.original_filepath).split(".")[0] # Remove file name extension
    save_path = os.path.join(original_file_dir, f"{original_filename}_translated_to_{args.target_lang}.json")
    logger.info(f"Saving translated original file to {save_path}")
    with open(save_path, "w") as f:
        json.dump(translated_original_data, f, indent=4, ensure_ascii=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_filepath", type=str, default="./data/Cspider/dev.json",
                        help = 'file path of the original file (for registing evaluator).')
    parser.add_argument("--target_lang", type=str, default="en",
                        help="The target language to translate. Default is english")

    args = parser.parse_args()
    main(args)