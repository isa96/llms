import os
from src.logger import get_console_logger
from src.api_news import download_news_api


# Initialize logger
logger = get_console_logger()

# Retrieve News API key from environment variables
try:
    NEWS_API_KEY = os.environ["NEWS_API_KEY"]
except KeyError:
    logger.error("Please set the environment variables NEWS_API_KEY")
    raise

url = f'https://eodhd.com/api/news?s=AAPL.US&offset=0&limit=10&api_token={NEWS_API_KEY}&fmt=json'
download_news_api(url=url)