import redis
import json
import hashlib
import numpy as np
import logging
import re
import time
import os

logger = logging.getLogger(__name__)

# Common abbreviation expansions for fair query comparison
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

# Regex to detect time-sensitive or highly personal queries that should not be cached
UNSUITABLE_CACHE_PATTERNS = [
    r'\bwhat\s+time\s+is\s+it\b',
    r'\bcurrent\s+time\b',
    r'\btoday(?:\'s)?\s+date\b',
    r'\bwhat\s+is\s+today\b',
    r'\bweather\s+(?:today|now|tomorrow)\b',
    r'\bstock\s+price\b',
    r'\blatest\s+news\b',
    r'\bwho\s+am\s+i\b',
    r'\bmy\s+(?:name|account|password|email|balance|location)\b',
]

def normalize_query(text: str) -> str:
    """Lowercase + expand common abbreviations so embeddings compare fairly."""
    text = text.lower().strip()
    for pattern, expansion in ABBREVIATIONS.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    # collapse extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_query_cacheable(text: str) -> bool:
    """Check if query is suitable for semantic caching."""
    q_lower = text.lower()
    for pat in UNSUITABLE_CACHE_PATTERNS:
        if re.search(pat, q_lower):
            return False
    return True

from hallucination.detector import get_embedder

class CacheModule:
    def __init__(self, threshold=None, persistence_file="cache_store.json"):
        import threading
        self._lock = threading.Lock()
        self._redis = None
        self._redis_offline = True
        self.model = get_embedder()
        
        # Adjustable threshold: default to env or 0.82 (validated for Gemini gemini-embedding-2)
        env_threshold = os.getenv("SEMANTIC_CACHE_THRESHOLD")
        if threshold is not None:
            self.threshold = float(threshold)
        elif env_threshold:
            self.threshold = float(env_threshold)
        else:
            self.threshold = 0.82
            
        self.cache_prefix = 'llm_cache:'
        self.persistence_file = persistence_file
        self._in_memory_cache = []  # List of entry dicts
        
        # Load persisted in-memory entries if available
        self._load_local_store()

        # Start connection check loop in background thread
        threading.Thread(target=self._check_redis_connection_loop, daemon=True).start()

    def _load_local_store(self):
        if self.persistence_file and os.path.exists(self.persistence_file):
            try:
                with open(self.persistence_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._in_memory_cache = data
                        logger.info(f"Loaded {len(data)} cached entries from {self.persistence_file}")
            except Exception as e:
                logger.warning(f"Could not load local cache persistence: {e}")

    def _save_local_store(self):
        if self.persistence_file:
            try:
                with open(self.persistence_file, 'w', encoding='utf-8') as f:
                    json.dump(self._in_memory_cache, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save local cache persistence: {e}")

    def _check_redis_connection_loop(self):
        while True:
            try:
                redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
                r = redis.Redis.from_url(redis_url, socket_timeout=0.5, socket_connect_timeout=0.5)
                r.ping()
                self._redis = r
                self._redis_offline = False
            except Exception:
                self._redis = None
                self._redis_offline = True
            time.sleep(60)

    def _get_redis(self):
        """Returns the background-established Redis instance, or None if offline."""
        if self._redis_offline or self._redis is None:
            return None
        return self._redis

    def _embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        """
        Embed text using Gemini embedder and guarantee 1D numpy array.

        task_type must be:
          - "RETRIEVAL_QUERY"    when embedding an incoming search query (get).
          - "RETRIEVAL_DOCUMENT" when embedding content to be stored (set).
        Mismatching these roles suppresses cosine similarity even for identical
        strings and is the root cause of false cache misses with Gemini embeddings.
        """
        raw = self.model.encode(normalize_query(text), task_type=task_type)
        arr = np.array(raw, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1)
        return arr

    def _cosine_similarity(self, a, b) -> float:
        a_arr = np.array(a, dtype=np.float32).reshape(-1)
        b_arr = np.array(b, dtype=np.float32).reshape(-1)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))

    def get(self, query: str, query_embedding=None) -> dict | None:
        """
        Lookup semantic cache.

        query_embedding: pre-computed RETRIEVAL_QUERY embedding (numpy array or list).
            If None, one is computed here. Do NOT pass a RETRIEVAL_DOCUMENT embedding;
            the asymmetric task_types are what give Gemini Embedding 2 its retrieval power.

        Returns a dict with hit metadata if found, or None if missed or uncacheable.
        """
        start_time = time.time()

        # Check cacheability
        if not is_query_cacheable(query):
            logger.info(f"Query '{query}' is time-sensitive or personal – bypassing cache.")
            return None

        # Always embed the lookup query with RETRIEVAL_QUERY task_type.
        # If a pre-computed RETRIEVAL_QUERY embedding is supplied, use it directly.
        if query_embedding is not None:
            query_vec = np.array(query_embedding, dtype=np.float32).reshape(-1)
        else:
            query_vec = self._embed(query, task_type="RETRIEVAL_QUERY")

        r = self._get_redis()
        
        # Try Redis first if available
        if r is not None:
            try:
                best_score = 0.0
                best_entry = None

                for key in r.scan_iter(f'{self.cache_prefix}*'):
                    raw = r.get(key)
                    if raw is None:
                        continue
                    entry = json.loads(raw)
                    cached_vec = np.array(entry['embedding'], dtype=np.float32).reshape(-1)
                    score = self._cosine_similarity(query_vec, cached_vec)
                    if score > best_score:
                        best_score = score
                        best_entry = entry

                lookup_latency = (time.time() - start_time) * 1000

                if best_entry and best_score >= self.threshold:
                    logger.info(f"Cache HIT (Redis) | similarity={best_score:.3f} >= {self.threshold} | query='{query}' matched='{best_entry.get('query')}'")
                    return {
                        'response': best_entry['response'],
                        'cache_hit': True,
                        'similarity': round(best_score, 4),
                        'matched_query': best_entry.get('query', ''),
                        'cache_source': 'redis',
                        'lookup_latency_ms': round(lookup_latency, 2),
                        'model_metadata': best_entry.get('model_metadata', {}),
                        'cached_at': best_entry.get('timestamp', 0)
                    }
                else:
                    logger.info(f"Cache MISS (Redis) | best_score={best_score:.3f} < {self.threshold} | query='{query}'")
            except redis.ConnectionError as e:
                logger.warning(f"Redis connection error during get: {e}")
                self._redis = None

        # Fallback to In-Memory Cache (thread-safe copy)
        best_score = 0.0
        best_entry = None
        with self._lock:
            cache_copy = list(self._in_memory_cache)
            
        for entry in cache_copy:
            cached_vec = np.array(entry['embedding'], dtype=np.float32).reshape(-1)
            score = self._cosine_similarity(query_vec, cached_vec)
            if score > best_score:
                best_score = score
                best_entry = entry

        lookup_latency = (time.time() - start_time) * 1000

        if best_entry and best_score >= self.threshold:
            logger.info(f"Cache HIT (In-Memory) | similarity={best_score:.3f} >= {self.threshold} | query='{query}' matched='{best_entry.get('query')}'")
            return {
                'response': best_entry['response'],
                'cache_hit': True,
                'similarity': round(best_score, 4),
                'matched_query': best_entry.get('query', ''),
                'cache_source': 'in_memory',
                'lookup_latency_ms': round(lookup_latency, 2),
                'model_metadata': best_entry.get('model_metadata', {}),
                'cached_at': best_entry.get('timestamp', 0)
            }
        else:
            logger.info(f"Cache MISS (In-Memory) | best_score={best_score:.3f} < {self.threshold} | query='{query}'")

        return None

    def set(self, query: str, response: str, doc_embedding=None, model_metadata: dict = None):
        """
        Store query-response in cache with rich metadata.

        doc_embedding: pre-computed RETRIEVAL_DOCUMENT embedding (numpy array or list).
            If None, one is computed here. This MUST be a RETRIEVAL_DOCUMENT embedding,
            not a RETRIEVAL_QUERY embedding, so that similarity scores are correct.
        """
        if not is_query_cacheable(query):
            return

        if doc_embedding is not None:
            embedding = doc_embedding.tolist() if hasattr(doc_embedding, 'tolist') else list(doc_embedding)
        else:
            embedding = self._embed(query, task_type="RETRIEVAL_DOCUMENT").tolist()

        entry = {
            'query': query,
            'normalized_query': normalize_query(query),
            'response': response,
            'embedding': embedding,
            'model_metadata': model_metadata or {},
            'timestamp': time.time()
        }

        # Save to In-Memory Cache
        with self._lock:
            self._in_memory_cache.append(entry)
            self._save_local_store()
        logger.info(f"Cache SET (In-Memory) | query='{query}'")

        # Save to Redis if available
        r = self._get_redis()
        if r is None:
            return
        try:
            key = self.cache_prefix + hashlib.md5(normalize_query(query).encode()).hexdigest()
            r.set(key, json.dumps(entry))
            logger.info(f"Cache SET (Redis) | query='{query}'")
        except redis.ConnectionError as e:
            logger.warning(f"Redis connection error during set: {e}")
            self._redis = None