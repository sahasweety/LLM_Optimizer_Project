import redis
import json
import hashlib
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

# ── Common abbreviation expansions ──────────────────────────────────────────
ABBREVIATIONS = {
    r'\bml\b':   'machine learning',
    r'\bai\b':   'artificial intelligence',
    r'\bdl\b':   'deep learning',
    r'\bnlp\b':  'natural language processing',
    r'\bnn\b':   'neural network',
    r'\bcnn\b':  'convolutional neural network',
    r'\brnn\b':  'recurrent neural network',
    r'\bllm\b':  'large language model',
    r'\brl\b':   'reinforcement learning',
    r'\bsvm\b':  'support vector machine',
    r'\bknn\b':  'k nearest neighbors',
    r'\bpca\b':  'principal component analysis',
    r'\bgpt\b':  'generative pre-trained transformer',
    r'\bapi\b':  'application programming interface',
    r'\bdb\b':   'database',
    r'\bos\b':   'operating system',
    r'\bui\b':   'user interface',
    r'\boop\b':  'object oriented programming',
}

def normalize_query(text: str) -> str:
    """Lowercase + expand common abbreviations so embeddings compare fairly."""
    text = text.lower().strip()
    for pattern, expansion in ABBREVIATIONS.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    # collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


from hallucination.detector import get_embedder

class CacheModule:
    def __init__(self, threshold=0.75):
        self._redis = None
        self.model = get_embedder()
        self.threshold = threshold          # set to 0.75 for a balanced trade-off
        self.cache_prefix = 'llm_cache:'
        self._in_memory_cache = []          # List of dicts: {'query': str, 'response': str, 'embedding': list}

    def _get_redis(self):
        """Lazy Redis connection – returns None if unavailable."""
        if self._redis is None:
            try:
                r = redis.Redis(host='localhost', port=6379, db=0,
                                socket_connect_timeout=3)
                r.ping()
                self._redis = r
                logger.info("Redis connected successfully.")
            except Exception as e:
                logger.warning(f"Redis not available: {e}")
                return None
        return self._redis

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([normalize_query(text)])[0]

    def _cosine_similarity(self, a, b) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def get(self, query: str):
        query_vec = self._embed(query)
        r = self._get_redis()
        
        if r is not None:
            try:
                best_score = 0.0
                best_entry = None

                for key in r.scan_iter(f'{self.cache_prefix}*'):
                    raw = r.get(key)
                    if raw is None:
                        continue
                    entry = json.loads(raw)
                    cached_vec = np.array(entry['embedding'])
                    score = self._cosine_similarity(query_vec, cached_vec)
                    if score > best_score:
                        best_score = score
                        best_entry = entry

                if best_entry and best_score >= self.threshold:
                    logger.info(f"Cache HIT (Redis) | similarity={best_score:.3f} | query='{query}'")
                    return {
                        'response': best_entry['response'],
                        'cache_hit': True,
                        'similarity': best_score
                    }
                else:
                    logger.info(f"Cache MISS (Redis) | best_score={best_score:.3f} | query='{query}'")
            except redis.ConnectionError as e:
                logger.warning(f"Redis connection lost: {e}")
                self._redis = None

        # Fallback to In-Memory Cache
        best_score = 0.0
        best_entry = None
        for entry in self._in_memory_cache:
            cached_vec = np.array(entry['embedding'])
            score = self._cosine_similarity(query_vec, cached_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self.threshold:
            logger.info(f"Cache HIT (In-Memory) | similarity={best_score:.3f} | query='{query}'")
            return {
                'response': best_entry['response'],
                'cache_hit': True,
                'similarity': best_score
            }
        else:
            logger.info(f"Cache MISS (In-Memory) | best_score={best_score:.3f} | query='{query}'")

        return None

    def set(self, query: str, response: str):
        embedding = self._embed(query).tolist()
        
        # Save to In-Memory Cache
        self._in_memory_cache.append({
            'query': query,
            'response': response,
            'embedding': embedding
        })
        logger.info(f"Cache SET (In-Memory) | query='{query}'")

        r = self._get_redis()
        if r is None:
            return
        try:
            key = self.cache_prefix + hashlib.md5(
                normalize_query(query).encode()).hexdigest()
            r.set(key, json.dumps({
                'query': query,
                'response': response,
                'embedding': embedding
            }))
            logger.info(f"Cache SET (Redis) | query='{query}'")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection lost: {e}")
            self._redis = None