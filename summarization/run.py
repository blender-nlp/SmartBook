import openai
from tqdm import tqdm
from copy import deepcopy
import argparse
import json
import os

def run_summarization(data, api_key):
    openai.api_key = api_key
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
            inp_text = inp_text + "You should cite the appropriate contexts where necessary. Each sentence in the summary should have a citation and the citations should be in the form of just the context number.\n\n"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=inp_text,
                temperature=0.7,
                max_tokens=250,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            data[cluster_index]["questions"][question]["summary"] = response["choices"][0]["text"].strip()
        
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

    args = parser.parse_args()

    claim_data = json.load(open(os.path.join(args.input_dir, "output_claims.json")))
    summaries = run_summarization(claim_data, args.openai_api_key)
    
    json.dump(summaries, open(os.path.join(args.output_dir, "output_summaries.json"), "w"), indent=4)

    