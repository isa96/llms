import os
import json 
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List

import requests
from src.logger import get_console_logger
from src.path import DATA_DIR

# Initialize logger
logger = get_console_logger()

@dataclass
class News:
    """
    Data class representing a news item with attributes such as title, content,
    sentiment, and date.
    """
    title: str
    content: str
    sentiment: str
    date: datetime

def get_batch_news(url: str) -> List[News]:
    """
    Fetches a batch of news items from the EODHD API and processes them to
    create a list of News objects.

    Args:
        url (str): The URL of the EODHD API.

    Returns:
        List[News]: A list of News objects containing news details.
    """
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        list_of_news = []
        for item in data:
            # Determine sentiment based on polarity
            if item["sentiment"]["polarity"] < 0:
                sentiment_value = "Negative"
            elif item["sentiment"]["polarity"] == 0:
                sentiment_value = "Neutral"
            else:
                sentiment_value = "Positive"

            list_of_news.append(
                News(
                    title=item["title"],
                    content=item["content"],
                    sentiment=sentiment_value,
                    date=item['date']
                )
            )
        return list_of_news
    else:
        logger.error("Request failed with status code:", response.status_code)
        return []

def save_to_json(news_list: List[News], filename: Path):
    """
    Saves a list of News objects to a JSON file.

    Args:
        news_list (List[News]): The list of News objects.
        filename (Path): The path to the JSON file to be saved.
    """
    news_data = [
        {
            "headline": news.title,
            "date": news.date,
            "content": news.content,
            "sentiment": news.sentiment
        }
        for news in news_list
    ]

    with open(filename, "w") as json_file:
        json.dump(news_data, json_file, indent=4)

def download_news_api(url: str) -> Path:
    """
    Downloads news from the EODHD API, processes and saves the data to a JSON
    file, and returns the path to the saved file.

    Args:
        url (str): The URL of the EODHD API.

    Returns:
        Path: The path to the saved JSON file.
    """
    logger.info("Downloading historical news...")
    list_of_news = get_batch_news(url=url)
    
    if list_of_news:
        logger.info("Successfully retrieved data")
        logger.info("Saving data to JSON...")
        path_to_file = DATA_DIR / "financial_news.json"
        save_to_json(list_of_news, path_to_file)
        logger.info(f"News data saved to {path_to_file}")
        return path_to_file
    else:
        logger.warning("No news data to save.")
        return None
