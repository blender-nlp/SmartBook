import glob
import os
import json
from nltk import word_tokenize
from copy import deepcopy
import re
import numpy as np
import torch
import argparse
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

def make_input_batch(source_batch_articles, tokenizer, max_input_length, max_title_length, max_article_length, use_title, device):
    
    source_batch = list()
    for articles in [source_batch_articles]:
        article_texts = list()
        for item in articles[:5]:
            if use_title:
                article_texts.append((" " + tokenizer.sep_token + " ").join([" ".join(word_tokenize(item["text"].split("\n")[0].strip())[:max_title_length]), " ".join(word_tokenize("\n".join(item["text"].split("\n")[1:]).strip())[:max_article_length])]))
            else:
                article_texts.append(" ".join(word_tokenize(item["text"])[:max_article_length]))                
        source_batch.append((" " + tokenizer.sep_token + " ").join(article_texts))

    source_toks = tokenizer.batch_encode_plus(source_batch,  max_length=max_input_length, truncation=True, pad_to_max_length=True)
    source_ids, source_mask = (
        torch.LongTensor(source_toks["input_ids"]).to(device),
        torch.LongTensor(source_toks["attention_mask"]).to(device),
    )
    model_inputs = {
        "input_ids": source_ids,
        "attention_mask": source_mask
    }

    return model_inputs


def generate_headline(args, clusters):
    if args.use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    print("Generating Headlines on device: ", device)
    
    model =  BartForConditionalGeneration.from_pretrained(args.headline_model_path)
    tokenizer = BartTokenizer.from_pretrained(args.headline_model_path)
    model.to(device)
    for key in tqdm(clusters):
        model_inputs = make_input_batch(clusters[key]["cluster_articles"], tokenizer=tokenizer, max_input_length=args.max_input_length, max_article_length=args.maximum_article_length, max_title_length=args.max_title_length, use_title=args.use_title, device=device)
        generated_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            min_length=args.min_generate_len,
            max_length=args.max_generate_len,
            do_sample=False,
            early_stopping=True,
            num_beams=args.beam_size,
            temperature=1.0,
            top_k=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
        clusters[key]["cluster_headline"] = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    return clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument("--use_gpu", action='store_true', help="whether to use the GPU")
    parser.add_argument("--run_headline", action='store_true', help="whether to headline generation")
    parser.add_argument("--headline_model_path", type=str, help="path to headline generator model")
    parser.add_argument("--input_dir", type=str, help="path to input directory")
    parser.add_argument("--output_dir", type=str, help="path to output directory")
    parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--maximum_article_length", type=int, default=150, help="Maximum length of each article to input")
    parser.add_argument("--min_generate_len", type=int, default=6, help="Minimum answer length while generation")
    parser.add_argument("--max_generate_len", type=int, default=40, help="Maximum answer length while generation")
    parser.add_argument("--use_title", action='store_true', help="whether to pass title as input")
    parser.add_argument("--max_title_length", type=int, default=40, help="Maximum title length when title is passed as input")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size while generation")

    args = parser.parse_args()

    clusters = json.load(open(os.path.join(args.output_dir, "output_clusters.json")))
    clusters = generate_headline(args, clusters)
    
    json.dump(clusters, open(os.path.join(args.output_dir, "output_headline.json"), "w"), indent=4)