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
        _shared_embedder = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
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

    def _do_single_call(self, query: str, model_info: dict) -> str:
        r = self.client.call('Answer factually.', query, model_info)
        return r['response']

    def _consistency_score(self, query: str, model_info: dict) -> float:
        """
        Runs two consistency checks in parallel background threads with a hard
        timeout so the caller never waits more than CONSISTENCY_TIMEOUT seconds.
        """
        f1 = _executor.submit(self._do_single_call, query, model_info)
        f2 = _executor.submit(self._do_single_call, query, model_info)
        
        responses = []
        try:
            r1 = f1.result(timeout=CONSISTENCY_TIMEOUT)
            responses.append(r1)
        except Exception as e:
            logger.warning(f"Consistency check call 1 failed or timed out: {e}")
            f1.cancel()
            
        try:
            r2 = f2.result(timeout=CONSISTENCY_TIMEOUT)
            responses.append(r2)
        except Exception as e:
            logger.warning(f"Consistency check call 2 failed or timed out: {e}")
            f2.cancel()

        if len(responses) < 2:
            return 0.3              # neutral fallback

        vecs = self.embedder.encode(responses)
        sim  = np.dot(vecs[0], vecs[1]) / (
               np.linalg.norm(vecs[0]) * np.linalg.norm(vecs[1]))

        inconsistency = float(1.0 - sim)
        return min(inconsistency * 1.5, 1.0)

    REFUSAL_PHRASES = [
        "cannot answer", "don't have access", "do not have access", 
        "cannot verify", "unable to provide", "as an ai", "sorry, but", 
        "cannot provide", "i do not know", "i don't know", "real-time data", 
        "my knowledge cutoff", "is not available", "cannot confirm"
    ]

    def _detect_refusal(self, response: str) -> bool:
        text = response.lower()
        return any(p in text for p in self.REFUSAL_PHRASES)

    def score(self, query: str, response: str, model_info: dict) -> dict:
        reasons = []
        
        # 1. Refusal Check
        if self._detect_refusal(response):
            return {
                'hallucination_score': 0.0,
                'risk_level': 'low',
                'explanation': 'Low Risk: The model correctly and safely refused to answer, avoiding hallucination.',
                'confidence_check': 0.0,
                'consistency_check': 0.0
            }
            
        # 2. Confidence / Hedging / Speculative Check
        conf_score = self._confidence_score(query, response)
        if conf_score > 0.4:
            reasons.append("Contains highly speculative phrasing or uncertainty markers.")
        elif conf_score > 0.15:
            reasons.append("Contains mild hedging language (e.g., 'probably', 'likely').")
            
        # 3. Consistency check (Semantic variance across multiple runs)
        consist_score = self._consistency_score(query, model_info)
        if consist_score > 0.4:
            reasons.append("High inconsistency detected between multiple model runs, suggesting potential fabrication.")
        elif consist_score > 0.15:
            reasons.append("Minor differences in detail detected between multiple model runs.")
            
        # Compute final score
        final = (conf_score * 0.45) + (consist_score * 0.55)
        final = min(final, 1.0)
        
        # Risk level determination
        if final > 0.5:
            risk = 'high'
            base_desc = "High Risk: The response has high semantic inconsistency across attempts and/or relies heavily on speculative statements."
        elif final > 0.25:
            risk = 'medium'
            base_desc = "Medium Risk: Minor inconsistency or mild speculation detected. Please cross-reference critical facts."
        else:
            risk = 'low'
            base_desc = "Low Risk: Response is semantically consistent and contains no speculative or uncertain language."
            
        # Build explanation
        if reasons:
            explanation = f"{base_desc} Details: " + " ".join(reasons)
        else:
            explanation = base_desc
            
        return {
            'hallucination_score': round(final, 3),
            'risk_level': risk,
            'explanation': explanation,
            'confidence_check': round(conf_score, 3),
            'consistency_check': round(consist_score, 3)
        }