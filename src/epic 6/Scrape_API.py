import requests
from bs4 import BeautifulSoup
import json
from googletrans import Translator
from transformers import pipeline
import os

# Function to translate text
def translate_text(text, dest_language):
    translator = Translator()
    try:
        translation = translator.translate(text, dest=dest_language)
        return translation.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return text

# Function to fetch articles from API
def fetch_articles(api_url, headers):
    response = requests.get(api_url, headers=headers)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        print(f"Failed to retrieve data from API: Status code {response.status_code}")
        return []

# Function to parse article content
def parse_article_content(url, headers):
    article_response = requests.get(url, headers=headers)
    if article_response.status_code == 200:
        soup = BeautifulSoup(article_response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        return ' '.join([p.get_text() for p in paragraphs])
    else:
        print(f"Failed to retrieve article at {url}: Status code {article_response.status_code}")
        return ""

def save_article_data(index, url, original_title, translated_title, article_text, sentiment):
    folder_path = 'C:\\Users\\Nicolas\\Documents\\Hogent Jaar 3\\Programming\\Data Engineering Project II\\github\\Data-Engineering-Project-II\\src\\epic 6\\API_Data'
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_path = os.path.join(folder_path, f'article_{index}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(f"URL: {url}\n")
            file.write(f"Original Title: {original_title}\n")
            file.write(f"Translated Title: {translated_title}\n")
            file.write(f"Sentiment: {sentiment}\n")
            # Uncomment below if you want to save the article text as well
            # file.write("Text:\n")
            # file.write(article_text)
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    api_url = "https://newsapi.org/v2/top-headlines?country=be&apiKey=87ca85a0f3a7468a9e89bb55ae3ef1c7"
    
    sentiment_pipeline = pipeline("sentiment-analysis")
    articles = fetch_articles(api_url, headers)
    print(f"Fetched {len(articles)} articles")

    for index, article in enumerate(articles):
        article_url = article.get("url")
        article_title = article.get("title", "No Title")
        translated_title = translate_text(article_title, 'nl')
        print(f"Translated title: {translated_title}")  

        if article_url:
            article_text = parse_article_content(article_url, headers)
            sentiment = sentiment_pipeline(translated_title)[0]
            print(f"Sentiment: {sentiment}")
            save_article_data(index, article_url, article_title, translated_title, article_text, sentiment)

if __name__ == "__main__":
    main()
