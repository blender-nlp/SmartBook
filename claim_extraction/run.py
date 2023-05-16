import json
from tqdm import tqdm
import requests
from newspaper import Article   
from copy import deepcopy
import re
from transformers import pipeline   
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import argparse
import os
import spacy
nlp = spacy.load("en_core_web_sm")

def get_sentences(text):
    sent_data = list()
    doc = nlp(text)
    for sent in doc.sents:
        item = dict()
        item["start_char"] =  int(sent.start_char) 
        item["end_char"] = int(sent.end_char) 
        item["sentence"] = sent.text
        sent_data.append(deepcopy(item))
    return sent_data

def get_expanded_corpus(data, api_key, start_date, end_date):
    url = "https://serpapi.com/search.json"
    output = dict()
    for cluster_index in tqdm(data):
        cluster = data[cluster_index]
        print("Chapter: ", cluster["cluster_headline"])
        news_output = list()
        try:
            query = {
                "q": cluster["cluster_headline"] + " before:" + end_date + " after:" + start_date,
                "location":"United States",
                "tbm":"nws",
                "api_key": api_key
            }
            response = requests.request("GET", url, params=query)
            newsapi_output = json.loads(response.text)    
            news_output.extend(newsapi_output["news_results"][:10])
        except Exception as e:
            print(e)
            continue

        assert len(news_output) > 0
        try:
            query = {
                "q": cluster["cluster_headline"] + " before:" + end_date + " after:" + start_date,
                "location":"United States",
                "tbm":"nws",
                "api_key": api_key,
                "start": len(news_output)
            }
            response = requests.request("GET", url, params=query)
            newsapi_output = json.loads(response.text)    
            news_output.extend(newsapi_output["news_results"][:5])
        except Exception as e:
            print(e)
            continue

        extracted_articles = list()
        
        for article in tqdm(news_output[:15]):
            try:
                cc_article = Article(article["link"])
                cc_article.download()
                cc_article.parse()
                extracted_item = dict()
                if cc_article.text.strip() != "":
                    extracted_item["cc_text"] = cc_article.text  
                    extracted_item["cc_title"] = cc_article.title
                    extracted_item["link"] = article["link"]
                    extracted_articles.append(deepcopy(extracted_item))
            except:
                continue
                
        print("Extracted Count: ", len(extracted_articles)) 
        output_item = dict()
        output_item["cluster_headline"] = cluster["cluster_headline"]
        output_item["cnn_article_titles"] = deepcopy(cluster["article_titles"])
        output_item["questions"] = deepcopy(cluster["picked_questions"])
        output_item["google_articles"] = deepcopy(extracted_articles)
        output[cluster_index] = deepcopy(output_item)    

    return output   

def combine_all_content(data, clusters):
    cluster_mapping = dict()
    for cluster_index in clusters:
        cluster = clusters[cluster_index]
        cluster_mapping[cluster["cluster_headline"]] = dict()
        for article in cluster["cluster_articles"]:
            if article["headline"].strip():
                cluster_mapping[cluster["cluster_headline"]][article["headline"]] = deepcopy(article)
    
    for cluster_index in data:
        cluster = data[cluster_index]
        data[cluster_index]["all_articles"] = list()
        assert cluster["cluster_headline"] in cluster_mapping
        for headline in cluster["cnn_article_titles"]:
            if headline.strip():
                assert headline in cluster_mapping[cluster["cluster_headline"]]
                article_item = dict()
                article_item["cc_title"] = cluster_mapping[cluster["cluster_headline"]][headline]["headline"]
                article_item["cc_text"] = "\n".join(cluster_mapping[cluster["cluster_headline"]][headline]["text"].split("\n")[1:])
                article_item["id"] = cluster_mapping[cluster["cluster_headline"]][headline]["id"]
                data[cluster_index]["all_articles"].append(deepcopy(article_item))

        id_prefix = "_".join(cluster["cluster_headline"].replace("-","").split())
        if "google_articles" in cluster:
            for article_index, article in enumerate(cluster["google_articles"]):
                if article["cc_title"].strip():
                    article["id"] = id_prefix + str(article_index)
                    data[cluster_index]["all_articles"].append(deepcopy(article))

        for article_index, article in enumerate(data[cluster_index]["all_articles"]):
            text = article["cc_text"]
            text = text.replace('\\xe2\\x80\\x9c', '"')
            text = text.replace('\\xe2\\x80\\x9d', '"')
            text = text.replace('\\xe2\\x80\\x98', "'")
            text = text.replace('\\xe2\\x80\\x99', "'")
            text = text.replace('\u201c', '"')
            text = text.replace('\u201d', '"')
            text = text.replace('\\xc2\\xa0', " ")
            text = text.replace('\\"', '\"')
            text = text.replace("\\'", "\'")
            text = text.replace("\\xe2\\x80\\x94", "-")
            text = text.replace("\\xe2\\x80\\x93", "-")
            
            text = json.loads(json.dumps(text.encode('utf-8').decode('unicode_escape').encode("ascii", "ignore").decode())).strip()
            text = re.sub(r'[ \t\r]*\n\n+[ \t\r]*', '\n\n', text)
            text = re.sub(r'[\t \r]+', ' ', text)
            text = "\n\n".join(re.split(r'\n\n+', text))
            data[cluster_index]["all_articles"][article_index]["cc_text"] = text
        if "google_articles" in cluster:
            del data[cluster_index]["google_articles"]
    
    return data

def run_claim_extraction(data, claim_model_path, device):
    nlp = pipeline('question-answering', model=claim_model_path, tokenizer=claim_model_path, device=device)
    for cluster_index in tqdm(data):
        cluster = data[cluster_index]
        print("Event: ", cluster["cluster_headline"])
        data[cluster_index]["qa_output"] = dict()
        for question in tqdm(cluster["questions"]):
            data[cluster_index]["qa_output"][question] = dict()
            for article in cluster["all_articles"]:
                if article["cc_text"].strip() != "":
                    qa_input =  {
                                'question': question,
                                'context': article["cc_text"]
                            }
                    result = nlp(qa_input)
                    data[cluster_index]["qa_output"][question][article["id"]] = deepcopy(result)
    
    return data

def run_verification(data, verification_model_path, device):
    tokenizer = RobertaTokenizer.from_pretrained(verification_model_path)
    model = RobertaForSequenceClassification.from_pretrained(verification_model_path)
    model = model.to(device)
    for cluster_index in tqdm(data):
        cluster = data[cluster_index]
        print("Event: ", cluster["cluster_headline"])
        article_sent_data = dict()
        for article in cluster["all_articles"]:
            article_sent_data[article["id"]] = deepcopy(get_sentences(article["cc_text"]))

        for question in tqdm(cluster["qa_output"]):
            for doc_id in cluster["qa_output"][question]:                
                sent_data = article_sent_data[doc_id]
                answer_sentence_item = None
                answer_sent_index = -1
                answer_start = cluster["qa_output"][question][doc_id]["start"]
                answer_end = cluster["qa_output"][question][doc_id]["end"]
                for sent_index, sent_item in enumerate(sent_data):
                    if sent_item["start_char"] <= answer_start and sent_item["end_char"] >= answer_end:
                        answer_sentence_item = deepcopy(sent_item)
                        answer_sent_index = sent_index
                        break
                if answer_sentence_item != None:
                    context_start_index = max(0, answer_sent_index-2)
                    context_end_index = min(len(sent_data), answer_sent_index+2)
                    context = ""
                    summ_context = ""
                    for sent_index in range(context_start_index, context_end_index):
                        context += " " + sent_data[sent_index]["sentence"]
                    for sent_index in range(context_start_index+1, context_end_index-1):
                        summ_context += " " + sent_data[sent_index]["sentence"]
                    data[cluster_index]["qa_output"][question][doc_id]["context"] = context
                    data[cluster_index]["qa_output"][question][doc_id]["sentence"] = answer_sentence_item["sentence"]
                    data[cluster_index]["qa_output"][question][doc_id]["doc_id"] = doc_id               
                    input_ids = tokenizer.encode(question, context, return_tensors='pt')
                    input_ids = input_ids.to(device)
                    logits = model(input_ids)[0]
                    answer_prob = logits.softmax(dim=1).detach().cpu().numpy()[0][1]
                    data[cluster_index]["qa_output"][question][doc_id]["rerank_score"] = float(answer_prob)
                else:
                    data[cluster_index]["qa_output"][question][doc_id]["rerank_score"] = 0.0
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument("--use_gpu", action='store_true', help="whether to use the GPU")
    parser.add_argument("--input_dir", type=str, help="path to input directory")
    parser.add_argument("--output_dir", type=str, help="path to output directory")
    parser.add_argument("--serp_api_key", type=str, default="", help="Key for Google News Search API")
    parser.add_argument("--start_date", type=str, help="News search start date in yyyy-mm-dd format")
    parser.add_argument("--end_date", type=str, help="News search end date in yyyy-mm-dd format")
    parser.add_argument("--claim_model_path", type=str, help="Path to zero-shot QA-based claim extraction model")
    parser.add_argument("--verification_model_path", type=str, help="Path to answer sentence verification model to use for claim verification")

    args = parser.parse_args()
    if args.serp_api_key:
        assert args.start_date and args.end_date

    if args.use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    print("Extracting claims on device: ", device)

    question_data = json.load(open(os.path.join(args.input_dir, "output_questions.json")))

    if args.serp_api_key:
        corpus_data = get_expanded_corpus(question_data, args.serp_api_key, args.start_date, args.end_date)
    else:
        corpus_data = dict()
        for cluster_index in tqdm(question_data):
            cluster = question_data[cluster_index]
            output_item = dict()
            output_item["cluster_headline"] = cluster["cluster_headline"]
            output_item["cnn_article_titles"] = deepcopy(cluster["article_titles"])
            output_item["questions"] = deepcopy(cluster["picked_questions"])
            corpus_data[cluster_index] = deepcopy(output_item)   

    headline_data = json.load(open(os.path.join(args.input_dir, "output_headline.json")))

    full_data = combine_all_content(corpus_data, headline_data)

    extracted_claims = run_claim_extraction(full_data, args.claim_model_path, device)
    final_claims = run_verification(extracted_claims, args.verification_model_path, device)
    
    json.dump(final_claims, open(os.path.join(args.output_dir, "output_claims.json"), "w"), indent=4)
