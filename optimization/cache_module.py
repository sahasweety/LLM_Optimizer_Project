import redis
import json
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

class CacheModule:
    def __init__(self, threshold=0.60):
        self.redis = redis.Redis(host='localhost', port=6379, db=0)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.cache_prefix = 'llm_cache:'

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0]

    def _cosine_similarity(self, a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str):
        query_vec = self._embed(query)
        for key in self.redis.scan_iter(f'{self.cache_prefix}*'):
            entry = json.loads(self.redis.get(key))
            cached_vec = np.array(entry['embedding'])
            score = self._cosine_similarity(query_vec, cached_vec)
            if score >= self.threshold:
                return {
                    'response': entry['response'],
                    'cache_hit': True,
                    'similarity': score
                }
        return None

    def set(self, query: str, response: str):
        embedding = self._embed(query).tolist()
        key = self.cache_prefix + hashlib.md5(query.encode()).hexdigest()
        self.redis.setex(key, 86400, json.dumps({
            'query': query,
            'response': response,
            'embedding': embedding
        }))