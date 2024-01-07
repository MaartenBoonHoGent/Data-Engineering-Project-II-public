import os
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from googletrans import Translator
from dotenv import load_dotenv
from sqlalchemy import func
from datetime import datetime
from sqlalchemy import text
import uuid
from sqlalchemy import create_engine, text

load_dotenv()

def scrape_translate_and_analyze_sentiment(url):
    response = requests.get(url)
    if response.status_code != 200:
        print("Error in fetching the webpage.")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    titles = soup.find_all(class_='card-title')

    translator = Translator()
    translated_titles = [translator.translate(title.text.strip(), dest='en').text for title in titles]

    return translated_titles

def create_conn():
    driver = os.getenv("DB_DRIVER")
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    trusted_connection = os.getenv("DB_TRUSTED_CONNECTION")

    return create_engine(
        f"mssql+pyodbc://{server}/{database}?trusted_connection={trusted_connection}&driver={driver}"
    )

url = 'https://lv.vlaanderen.be/nieuws'
translated_data = scrape_translate_and_analyze_sentiment(url)

try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    sentiment_results = sentiment_pipeline(translated_data)
except Exception as e:
    print(f"Error during sentiment analysis: {e}")
    sentiment_results = []

engine = create_conn()
table_name = 'sentiment'

with engine.connect() as connection:
    
    for title, result in zip(translated_data, sentiment_results):
        sentiment_score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        random_id = str(uuid.uuid4())

        insert_statement = text(
            f"INSERT INTO {table_name} (id, sentiment, date, crm_activiteits_id) VALUES (:id, :sentiment, :current_date, :crm_activiteits_id)"
        )

        params = {
            'id': random_id,
            'sentiment': sentiment_score,
            'current_date': current_date,
            'crm_activiteits_id': '808F7E6B-1A62-E111-8F14-00505680000A'
        }


        connection.execute(insert_statement, params)
        connection.commit()

print("Sentiment analysis results saved to the database.")
