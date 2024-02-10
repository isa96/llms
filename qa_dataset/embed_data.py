import hashlib
from tqdm import tqdm
from pydantic import BaseModel
from typing import Dict, Optional, List

from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars
from unstructured.staging.huggingface import chunk_by_attention_window

from src.path import DATA_DIR
from src.logger import get_console_logger
from src.vector_db import get_qdrant_client, init_collection
from src.transformers import MODEL, TOKENIZER


NEWS_FILE = DATA_DIR / "financial_news.json"
QDRANT_COLLECION_NAME = "financial_news"
QDRANT_VECTOR_SIZE = 384

LOGGER = get_console_logger(__name__)

qdrant_client = get_qdrant_client()
qdrant_client = init_collection(
    qdrant_client=qdrant_client, 
    collection_name=QDRANT_COLLECION_NAME,
    vector_size=QDRANT_VECTOR_SIZE
)

class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list] = []
    chunks: Optional[list] = []
    embeddings: Optional[list] = []


def parse_document(_data: Dict) -> Document:
    """
    Parse the input data and create a Document object.
    
    Args:
        _data (Dict): Input data containing content, headline, and date.
    Returns:
        Document: The parsed document with cleaned text and metadata.
    """
    document_id = hashlib.md5(_data['content'].encode()).hexdigest()
    document = Document(id=document_id)

    article_elements = partition_html(text=_data['content'])
    _data['content'] = clean_non_ascii_chars(replace_unicode_quotes(clean(" ".join([str(x) for x in article_elements]))))
    _data['headline'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['headline'])))
    
    document.text = [_data['headline'], _data['content']]
    document.metadata['headline'] = _data['headline']
    document.metadata['date'] = _data['date']
    return document


def chunk(document: Document) -> Document:
    """
    Split the document into chunks using attention windows.
    Note: LangChain provides a `RecursiveCharacterTextSplitter` class that does that if required.
    
    Args:
        document (Document): Input document.
    Returns:
        Document: The document with chunks.
    """
    chunks = []
    for text in document.text:
        chunks += chunk_by_attention_window(
            text=text,
            tokenizer=TOKENIZER,
            max_input_size=QDRANT_VECTOR_SIZE
        )
    document.chunks = chunks
    return document

def embeddings(document: Document) -> Document:
    """
    Generate embeddings for the document text.

    Args:
        document (Document): Input document.
    Returns:
        Document: The document with embeddings.
    """
    for text in document.text:
        input_ = TOKENIZER(
            text=text,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=QDRANT_VECTOR_SIZE
        )

        result = MODEL(**input_)
        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
        lst = embeddings.flatten().tolist()
        document.embeddings.append(lst)
    return document


def build_payload(document: Document) -> List:
    """
    Build payloads for each chunk in the document.

    Args:
        document (Document): Input document.
    Returns:
        Tuple[List, List]: IDs and payloads for each chunk.
    """
    payloads = []
    ids = []
    for chunk in document.chunks:
        payload = document.metadata
        payload.update({"text": chunk})
        chunk_id = hashlib.md5(chunk.encode()).hexdigest()
        ids.append(chunk_id)
        payloads.append(payload)
    return ids, payloads


def push_document_to_qdrant(document: Document) -> None:
    """
    Push the document and its embeddings to the 
    Qdrant index.
    
    Args:
        document (Document): Input document.
    """
    
    from qdrant_client.models import PointStruct

    ids, _payloads = build_payload(document)

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECION_NAME,
        points=[
            PointStruct(
                id=idx,
                vector=vector,
                payload=_payload
            )
            for idx, vector, _payload in zip(ids, document.embeddings, _payloads)
        ]
    )


def proses_document(_data: Dict) -> None:
    """
    Process the input data by parsing, chunking,
    generating embeddings, and pushing to Qdrant.

    Args:
        _data (Dict): Input data containing content, headline, and date.
    """
    doc = parse_document(_data)
    chunk = chunk(doc)
    embedding = embeddings(chunk)
    push_document_to_qdrant(embedding)

    return embedding


def embed_news_into_qdrant(news_data: List[Dict], n_processes: int = 1) -> None:
    """
    Embed news data into Qdrant index,
    either sequentially or in parallel.

    Args:
        news_data (List[Dict]): List of news data dictionaries.
        n_processes (int): Number of processes for parallel execution (default is 1).
    """
    results = []
    if n_processes == 1:
        # sequential
        for _data in tqdm(news_data):
            result = proses_document(_data)
            results.append(result)
    
    else:
        # parallel
        import multiprocessing

        # Create a multiprocessing Pool
        with multiprocessing.Pool(processes=n_processes) as pool:
            # Use tqdm to create a progress bar
            results = list(tqdm(pool.imap(proses_document, news_data),
                                total=len(news_data),
                                desc="Processing",
                                unit="news"))


if __name__ == "__main__":

    import json
    with open(NEWS_FILE, 'r') as json_file:
        news_data = json.load(json_file)
    
    embed_news_into_qdrant(
        news_data=news_data,
        n_processes=1
    )