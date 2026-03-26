import numpy as np
from sentence_transformers import SentenceTransformer

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
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def _confidence_score(self, query: str, response: str) -> float:
        text = response.lower()
        query_lower = query.lower()

        uncertain_hits = sum(1 for p in self.UNCERTAIN_PHRASES if p in text)
        speculative_hits = sum(1 for t in self.SPECULATIVE_TOPICS if t in query_lower or t in text)

        confidence = min(uncertain_hits * 0.12, 0.5)
        speculative = min(speculative_hits * 0.15, 0.5)

        return min(confidence + speculative, 0.8)

    def _consistency_score(self, query: str, model_info: dict) -> float:
        responses = []
        for _ in range(3):
            r = self.client.call('Answer factually.', query, model_info)
            responses.append(r['response'])

        vecs = self.embedder.encode(responses)
        pairwise = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sim = np.dot(vecs[i], vecs[j]) / (
                    np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]))
                pairwise.append(sim)

        avg_sim = np.mean(pairwise)
        inconsistency = float(1.0 - avg_sim)
        return min(inconsistency * 1.5, 1.0)

    def score(self, query: str, response: str, model_info: dict) -> dict:
        conf = self._confidence_score(query, response)
        consist = self._consistency_score(query, model_info)

        final = (conf * 0.45) + (consist * 0.55)
        final = min(final, 1.0)

        return {
            'hallucination_score': round(final, 3),
            'risk_level': 'high' if final > 0.5 else 'medium' if final > 0.25 else 'low',
            'confidence_check': round(conf, 3),
            'consistency_check': round(consist, 3)
        }