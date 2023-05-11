import openai
from openai.error import RateLimitError
import backoff
import json
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
from copy import deepcopy
from sentence_transformers import CrossEncoder
import random
import numpy as np
import argparse
import os

@backoff.on_exception(backoff.expo, RateLimitError)
def generate_questions(data, api_key):
    openai.api_key = api_key
    for cluster_index in tqdm(data):
        item = data[cluster_index]
        articles = item["cluster_articles"]
        headline = item["cluster_headline"]
        input = "Chapter Name: " + headline + " during the Russia-Ukraine crisis\n\nContext:\n"
        non_null_texts = list()
        for article in articles:
            if article["text"].strip() != "":
                non_null_texts.append(article["text"])
        
        for index, article in enumerate(non_null_texts):
            text = " ".join(article.split("\n")[1:]).strip()
            input += str(index+1) + ") " + text.strip() + "\n"
        
        input += "\n\nWhat are some strategic questions for the chapter about " + headline + "?"

        data[cluster_index]["questions"] = list()

        for _ in range(0,3):
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=input,
                temperature=0.7,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            questions = response["choices"][0]["text"].strip()
            data[cluster_index]["questions"].append(questions)
        
        data[cluster_index]["article_titles"] = list()
        for article in item["cluster_articles"]:
            data[cluster_index]["article_titles"].append(article["text"].split("\n")[0].strip())
        data[cluster_index]["question_sets"] = list()
        for question in item["questions"]:
            data[cluster_index]["question_sets"].append(question.split("\n"))
        del data[cluster_index]["cluster_articles"]
        del data[cluster_index]["questions"]

    return data

def expand_questions(data, expand_question_model_path):
    tokenizer = T5Tokenizer.from_pretrained(expand_question_model_path)
    model = T5ForConditionalGeneration.from_pretrained(expand_question_model_path)
    for cluster_index in tqdm(data):
        cluster = data[cluster_index]
        title = cluster["cluster_headline"]
        data[cluster_index]["expanded_questions"] = list()
        for question_set in cluster["question_sets"]:
            expand_questions = list()
            question_base = " ".join(question_set[0].split()[1:])
            expand_questions.append(question_base)

            for question in question_set[1:]:
                context = title + " ||| " + question_base + " ||| " + " ".join(question.split()[1:])
                input_ids = tokenizer(context,return_tensors="pt").input_ids
                outputs = model.generate(input_ids, max_length=100)
                question_new = tokenizer.decode(outputs[0], skip_special_tokens=True)
                expand_questions.append(question_new)
            data[cluster_index]["expanded_questions"].append(deepcopy(expand_questions))

    return data

def remove_duplicates(data, duplicate_question_model_path, threshold):
    model = CrossEncoder(duplicate_question_model_path)
    for cluster_index in tqdm(data):
        cluster = data[cluster_index]
        print("Title: ", cluster["cluster_headline"])
        print("\n")
        all_questions = list()
        for s in cluster["expanded_questions"]:
            all_questions.extend(s)
        qset = [all_questions[0]]
        for question in all_questions[1:]:
            q_list = [(q, question) for q in qset]
            scores = model.predict(q_list)
            max_si = np.argmax(scores)
            if np.max(scores) < threshold:
                qset.append(question)
        data[cluster_index]["unique_questions"] = deepcopy(qset)
        qset = qset[1:]
        random.shuffle(qset)
        data[cluster_index]["picked_questions"] = list()
        data[cluster_index]["picked_questions"].append(data[cluster_index]["unique_questions"][0])
        data[cluster_index]["picked_questions"].extend(qset[:4])
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument("--openai_api_key", type=str, help="Open AI Key")
    parser.add_argument("--duplicate_threshold", type=float, default=0.70, help="Threshold to use for removing duplicate questions")
    parser.add_argument("--expand_question_model_path", type=str, help="Path to question expansion model")
    parser.add_argument("--duplicate_question_model_path", type=str, help="Path to duplicate question detection model")
    parser.add_argument("--output_dir", type=str, help="path to output directory")
    parser.add_argument("--input_dir", type=str, help="path to input directory")

    args = parser.parse_args()

    headline_data = json.load(open(os.path.join(args.input_dir, "output_headline.json")))
    questions = generate_questions(headline_data, args.openai_api_key)
    expanded_questions = expand_questions(questions, args.expand_question_model_path)
    final_questions = remove_duplicates(expanded_questions, args.duplicate_question_model_path, args.duplicate_threshold)
    
    json.dump(final_questions, open(os.path.join(args.output_dir, "output_questions.json"), "w"), indent=4)