Before running the code, you need to install BeautifulSoup library to allow for scraping of the articles:
```
pip install beautifulsoup4
```

To create SmartBook reports for the Ukraine-Russia crisis, you need to first scrape the CNN daily coverage articles. These will be clustered together within SmartBook to identify the major events.

You can run the code to scrape all of CNN's daily coverage within a given time period (*start_date* and *end_date* in `mm-dd-yyyy` format):

```
python cnn_ukraine_crawler_json.py --start_date <start_date_here> --end_date <end_date_here> --output_dir <output_dir_here>
```

The above code creates raw text files in the `output_dir` with each article having it's own file and file name corresponding to the article ID. The `output_dir` used here should be used as the `input_dir` when running the SmartBook code.