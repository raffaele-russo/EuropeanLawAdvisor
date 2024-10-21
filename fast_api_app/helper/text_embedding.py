"""
Module containing the class used to generate the embeddings 
from both the documents and the query
"""
from typing import List
from sentence_transformers import SentenceTransformer

class TextEmbedding:
    """Class responsible for the embeddings of one or a list of documents"""
    def __init__(self, model):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        "Embed a single document"
        return self.model.encode([query]).tolist()[0]
