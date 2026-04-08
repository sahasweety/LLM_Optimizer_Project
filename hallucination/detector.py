import numpy as np
from sentence_transformers import SentenceTransformer
import logging
import concurrent.futures

logger = logging.getLogger(__name__)

# Shared embedder instance (loaded once, reused everywhere)
_shared_embedder = None

# Thread pool for non-blocking consistency checks
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Timeout (seconds) for the entire consistency check
CONSISTENCY_TIMEOUT = 10

def get_embedder():
    global _shared_embedder
    if _shared_embedder is None:
        logger.info("Loading SentenceTransformer model (first use)...")
        _shared_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _shared_embedder


class HallucinationDetector:
    UNCERTAIN_PHRASES = [
        'i think', 'i believe', 'i am not sure', 'might be',
        'possibly', 'perhaps', 'not certain', 'i cannot verify',
        'i may be wrong', 'approximately', 'roughly', 'around',
        'estimated', 'likely', 'probably', 'unclear', 'uncertain',
        'i don\'t know', 'hard to say', 'difficult to say',
        'as of my knowledge', 'as of my last update', 'may have changed',
        'i cannot confirm', 'not sure', 'might have', 'could be',
        'as far as i know', 'to my knowledge', 'i\'m not certain',
        'i cannot provide exact', 'exact figures', 'precise number',
        'i don\'t have access', 'i cannot access', 'real-time',
        'current data', 'latest data', 'up to date'
    ]

    SPECULATIVE_TOPICS = [
        'stock price', 'stock market', 'predict', 'prediction', 'forecast',
        'future', '2025', '2026', '2027', 'salary', 'exact number',
        'exact salary', 'net worth', 'revenue', 'private meeting',
        'nobel prize', 'will be', 'will release', 'will happen',
        'exact cost', 'exact date', 'parameters', 'training cost'
    ]

    def __init__(self, llm_client):
        self.client = llm_client
        # Don't load model here — use shared lazy instance

    @property
    def embedder(self):
        return get_embedder()

    def _confidence_score(self, query: str, response: str) -> float:
        text        = response.lower()
        query_lower = query.lower()

        uncertain_hits   = sum(1 for p in self.UNCERTAIN_PHRASES if p in text)
        speculative_hits = sum(1 for t in self.SPECULATIVE_TOPICS
                               if t in query_lower or t in text)

        confidence  = min(uncertain_hits  * 0.12, 0.5)
        speculative = min(speculative_hits * 0.15, 0.5)

        return min(confidence + speculative, 0.8)

    def _do_consistency_check(self, query: str, model_info: dict) -> float:
        """
        Makes 2 extra LLM calls (reduced from 3) and computes semantic
        consistency between them.  Runs inside a thread-pool future so
        it never blocks the HTTP response path beyond CONSISTENCY_TIMEOUT.
        """
        responses = []
        for _ in range(2):          # reduced from 3 → 2 calls
            try:
                r = self.client.call('Answer factually.', query, model_info)
                responses.append(r['response'])
            except Exception as e:
                logger.warning(f"Consistency check call failed: {e}")

        if len(responses) < 2:
            return 0.3              # neutral fallback

        vecs = self.embedder.encode(responses)
        sim  = np.dot(vecs[0], vecs[1]) / (
               np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))

        inconsistency = float(1.0 - sim)
        return min(inconsistency * 1.5, 1.0)

    def _consistency_score(self, query: str, model_info: dict) -> float:
        """
        Runs the consistency check in a background thread with a hard
        timeout so the caller never waits more than CONSISTENCY_TIMEOUT
        seconds.  Returns 0.3 (neutral) on timeout or error.
        """
        future = _executor.submit(self._do_consistency_check, query, model_info)
        try:
            return future.result(timeout=CONSISTENCY_TIMEOUT)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning("Consistency check timed out — using neutral score 0.3")
            return 0.3
        except Exception as e:
            logger.warning(f"Consistency check error: {e} — using neutral score 0.3")
            return 0.3

    def score(self, query: str, response: str, model_info: dict) -> dict:
        conf    = self._confidence_score(query, response)
        consist = self._consistency_score(query, model_info)

        final = (conf * 0.45) + (consist * 0.55)
        final = min(final, 1.0)

        return {
            'hallucination_score': round(final, 3),
            'risk_level':          'high'   if final > 0.5  else
                                   'medium' if final > 0.25 else 'low',
            'confidence_check':    round(conf, 3),
            'consistency_check':   round(consist, 3)
        }