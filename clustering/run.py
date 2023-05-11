import glob
import os
import json
from nltk import word_tokenize
from copy import deepcopy
import re
# import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.neighbors import NearestCentroid
import argparse
from tqdm import tqdm

def process_text(text, stop_words):
    text = text.replace('\\xe2\\x80\\x9c', '"')
    text = text.replace('\\xe2\\x80\\x9d', '"')
    text = text.replace('\\xe2\\x80\\x98', "'")
    text = text.replace('\\xe2\\x80\\x99', "'")
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = text.replace('\u2019', "'")
    text = text.replace('\\xc2\\xa0', " ")
    text = text.replace('\\"', '\"')
    text = text.replace("\\'", "\'")
    text = text.replace("\\xe2\\x80\\x94", "-")
    text = text.replace("\\xe2\\x80\\x93", "-")
    try:
        text = json.loads(json.dumps(text.encode('utf-8').decode('unicode_escape').encode("ascii", "ignore").decode())).strip()
    except:
        print("Error processing file")
    text = re.sub(r'[ \t\r]*\n\n+[ \t\r]*', '\n\n', text)
    text = re.sub(r'[\t \r]+', ' ', text)
    text = "\n\n".join(re.split(r'\n\n+', text))

    text = text.replace("[^a-zA-Z#]", " ").lower()
    text = " ".join([i for i in text.split() if i not in stop_words])
    return text

def run_clustering(args):
    documents = list()
    fpaths = glob.glob(os.path.join(args.input_dir, "*.txt"))
    stop_words = stopwords.words("english")
    print("Processing ", len(fpaths), " documents")
    for fpath in tqdm(fpaths):
        fname = os.path.basename(fpath)
        text = open(fpath).read().strip()
        text = process_text(text, stop_words)
        document = dict()
        document["id"] = fname
        document["text"] = text
        documents.append(deepcopy(document))    

    texts = [item["text"] for item in documents]
    vectorizer_text = TfidfVectorizer(
        stop_words=stop_words,
        max_features=args.max_features,
        max_df=0.5,
        use_idf=True,
        ngram_range=(1, 3),
    )
    print("Featurizing text")
    X_text = vectorizer_text.fit_transform(texts)

    print("Running Clustering")
    clustering_text = AgglomerativeClustering(
        distance_threshold=args.distance_threshold, n_clusters=None
    ).fit(X_text.toarray())
    y_text = clustering_text.labels_

    clf = NearestCentroid()
    clf.fit(X_text, y_text)
    print("Clustering done")

    X_text_list = [list(x_feature) for x_feature in X_text]
    centroids_list = [list(centroid_feature) for centroid_feature in clf.centroids_]

    centroid_indices = [(i in centroids_list) for i in X_text_list]
    output = dict()

    for item, cluster_id, is_centroid in zip(documents, y_text, centroid_indices):
        if is_centroid:
            if str(cluster_id) not in output:
                output[str(cluster_id)] = dict()
                output[str(cluster_id)]["cluster_articles"] = list()
            output[str(cluster_id)]["cluster_articles"].append(item)

    for item, cluster_id in zip(documents, y_text):
        if not is_centroid:
            if str(cluster_id) not in output:
                output[str(cluster_id)] = dict()
                output[str(cluster_id)]["cluster_articles"] = list()
            output[str(cluster_id)]["cluster_articles"].append(item)

    print("Num Clusters found: ", len(output))

    final_clusters = dict()
    for key in output:
        if len(output[key]["cluster_articles"]) >= 4:
            final_clusters[key] = deepcopy(output[key])

    print("Num Clusters left after pruning: ", len(final_clusters))
    return final_clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument("--input_dir", type=str, help="path to input directory")
    parser.add_argument("--output_dir", type=str, help="path to output directory")
    parser.add_argument("--max_features", type=int, default=3000, help="max features to use for TF_IDF vectorizer")
    parser.add_argument("--distance_threshold", type=float, default=1.25, help="distance threshold to control number of clusters")

    args = parser.parse_args()

    clusters = run_clustering(args)
    json.dump(clusters, open(os.path.join(args.output_dir, "output_clusters.json"), "w"), indent=4)