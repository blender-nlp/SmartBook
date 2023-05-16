import os
import urllib
from bs4 import BeautifulSoup
from urllib.request import urlopen
import json
from collections import defaultdict
from PIL import Image
import requests
import argparse
from datetime import date, timedelta, datetime

parser = argparse.ArgumentParser(description='Parser')
parser.add_argument("--output_dir", type=str, help="output directory")
parser.add_argument("--start_date", type=str, help="News search start date in mm-dd-yyyy format")
parser.add_argument("--end_date", type=str, help="News search end date in mm-dd-yyyy format")
args = parser.parse_args()
output_dir=args.output_dir

date_start = datetime.strptime(args.start_date, '%m-%d-%Y').date()
date_end = datetime.strptime(args.end_date, '%m-%d-%Y').date()
delta = date_end - date_start   # returns timedelta

date_list = list()
os.makedirs(output_dir, exist_ok=True)

for i in range(delta.days + 1):
    day = date_start + timedelta(days=i)
    date_list.append(day.strftime('%m-%d-%y'))

for date_str in date_list:
    url_list = 'https://www.cnn.com/europe/live-news/russia-ukraine-war-news-%s/index.html' % date_str

    html = urlopen(url_list).read()
    soup = BeautifulSoup(html, 'html.parser')
    find_articles = soup.findAll('script', {"id":"liveBlog-schema"})
    if find_articles is not None:
        for article_list in find_articles:
            site_json=json.loads(article_list.text)

            for blog in site_json['liveBlogUpdate']:
                if 'articleBody' not in blog:
                    continue
                
                headline = blog['headline']
                url_article = blog['url']
                file_name = '%s' % url_article.split('/')[-1]
                datePublished = blog['datePublished'][:10]
                article_sum = blog['articleBody']
                image = blog['image']
                
                with open(os.path.join(output_dir, file_name+'.txt'), 'w') as writer:
                    if article_sum.startswith('Our live coverage of the'):
                        continue
                    if not article_sum.strip():
                        continue
                    if headline is not None:
                        writer.write(headline)
                        writer.write('\n')          
                    writer.write(article_sum)