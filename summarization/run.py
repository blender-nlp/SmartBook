import openai
from tqdm import tqdm
from copy import deepcopy
import argparse
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai.error import RateLimitError
import backoff

@backoff.on_exception(backoff.expo, RateLimitError)
def get_summary_from_openai(prompt):
    chatgpt_output = openai.ChatCompletion.create(
                model="gpt-4",
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
    summary = chatgpt_output["choices"][0]["message"]["content"].strip()
    return summary

def get_summary_from_llama(model, tokenizer, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=256, do_sample=True, top_p=1,temperature=0.7)
    summary = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    return summary

def run_summarization(data, args):
    if args.openai_api_key != None:
        print("Using GPT-4")
        openai.api_key = args.openai_api_key
    else:
        print("Using Llama-2")
        tokenizer = AutoTokenizer.from_pretrained(args.generation_model_path)
        model = AutoModelForCausalLM.from_pretrained(args.generation_model_path)
        model.to(device='cuda')
    for cluster_index in tqdm(data):
        cluster = data[cluster_index]
        print("Event: ", cluster["cluster_headline"])
        data[cluster_index]["questions"] = dict()
        for question in tqdm(cluster["qa_output"]):
            # time.sleep(3)
            data[cluster_index]["questions"][question] = dict()
            data[cluster_index]["questions"][question]["claims"] = list()
            results = sorted(cluster["qa_output"][question].items(), key=lambda item: item[1]["rerank_score"], reverse=True)[:5]
            for answer in results:
                data[cluster_index]["questions"][question]["claims"].append(deepcopy(answer[1]))
            inp_text = "You are given a set of contexts below:\n\n"
            inp_text = "\n".join([str(index+1) + ") " + i["context"] for index, i in enumerate(data[cluster_index]["questions"][question]["claims"])])
            inp_text = inp_text + "\n\nUsing the information above, write a coherent summary for " + question + "\n"
            inp_text = inp_text + "You should cite the appropriate contexts where necessary. Always cite for any factual claim. When citing several contexts, use [1][2][3]. Cite at least one context in each sentence..\n\n"

            if args.openai_api_key != None:
                summary = get_summary_from_openai(inp_text)
            else:
                summary = get_summary_from_llama(model, tokenizer, inp_text)
            
            data[cluster_index]["questions"][question]["summary"] = summary

        urls_mapping = {item["id"]:item["link"] for item in cluster["all_articles"]}
        for question in cluster["questions"]:
            for claim_index, claim_item in enumerate(cluster["questions"][question]["claims"]):
                data[cluster_index]["questions"][question]["claims"][claim_index]["link"] = urls_mapping[claim_item["doc_id"]]
        del data[cluster_index]["qa_output"]
        del data[cluster_index]["cnn_article_titles"]
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument("--openai_api_key", type=str, help="Open AI Key")
    parser.add_argument("--output_dir", type=str, help="path to output directory")
    parser.add_argument("--input_dir", type=str, help="path to input directory")
    parser.add_argument("--generation_model_path", type=str, help="Path to question generation model")

    args = parser.parse_args()
    assert args.openai_api_key != None or args.generation_model_path != None

    claim_data = json.load(open(os.path.join(args.input_dir, "output_claims.json")))
    summaries = run_summarization(claim_data, args.openai_api_key)
    
    json.dump(summaries, open(os.path.join(args.output_dir, "output_summaries.json"), "w"), indent=4)

    