from sentence_transformers import SentenceTransformer
from typing import List

class TextEmbedding:
        def __init__(self, model):
            self.model = SentenceTransformer(model, trust_remote_code=True)

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            print("texts: ",texts)
            return [self.model.encode(t).tolist() for t in texts]
        
        def embed_query(self, query: str) -> List[float]:
            print("query: ",query)
            return self.model.encode([query]).tolist()[0]
