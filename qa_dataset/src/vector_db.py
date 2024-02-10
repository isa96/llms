import os
from typing import Dict, List
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams

"""
Note: for experimental material I will use docker, 
before going to the streamming process
https://qdrant.tech/documentation/quick-start/
"""


def get_qdrant_client() -> QdrantClient:
    """
    Get an instance of the QdrantClient.

    Returns:
        QdrantClient: An instance of the QdrantClient.
    """
    client = QdrantClient("localhost", port=6333)
    return client

def init_collection(
        collection_name: str,
        vector_size: int,
        qdrant_client: QdrantClient
) -> QdrantClient:
    """
    Initialize or recreate a Qdrant collection.

    Args:
        collection_name (str): The name of the collection.
        vector_size (int): The size of vectors in the collection.
        qdrant_client (QdrantClient): An instance of the QdrantClient.

    Returns:
        QdrantClient: The QdrantClient instance with the specified collection initialized or recreated.
    """
    try:
        qdrant_client.get_collection(collection_name=collection_name)
    except (UnexpectedResponse, ValueError):
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
    return qdrant_client
