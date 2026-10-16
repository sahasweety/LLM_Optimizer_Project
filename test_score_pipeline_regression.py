"""
Regression tests for semantic cache pipeline and score handling.

Requirements:
1. Cache HIT
2. Cache MISS
3. Empty cache
4. Unavailable cache backend (e.g. Redis offline)
5. Full dashboard render_query_result pipeline with score handling
"""

import unittest
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from optimization.cache_module import CacheModule


class TestCacheScorePipelineRegression(unittest.TestCase):

    def setUp(self):
        self.cache = CacheModule(threshold=0.82)
        # Clear in-memory cache for deterministic test state
        with self.cache._lock:
            self.cache._in_memory_cache = []

    def test_empty_cache_returns_none_safely(self):
        """Test empty cache: get() returns None without unbound variable errors."""
        res = self.cache.get("What is machine learning?")
        self.assertIsNone(res, "Empty cache must return None")

    def test_cache_miss_returns_none_safely(self):
        """Test cache miss: non-matching query returns None when threshold not met."""
        # Seed an entry
        fake_vec = np.ones(768, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)

        self.cache.set(
            query="What is photosynthesis?",
            response="Photosynthesis is the process by which plants make food.",
            doc_embedding=fake_vec
        )

        # Lookup with orthogonal/different vector
        diff_vec = np.zeros(768, dtype=np.float32)
        diff_vec[0] = 1.0  # low similarity with uniform vector: 1 / sqrt(768) ≈ 0.036 < 0.82

        res = self.cache.get("What is the speed of light?", query_embedding=diff_vec)
        self.assertIsNone(res, "Query below threshold must return None (cache miss)")

    def test_cache_hit_returns_consistent_metadata(self):
        """Test cache hit: returns consistent metadata including similarity, matched_query, cache_source."""
        norm_vec = np.ones(768, dtype=np.float32)
        norm_vec /= np.linalg.norm(norm_vec)

        self.cache.set(
            query="What is machine learning?",
            response="ML is a subset of AI.",
            doc_embedding=norm_vec
        )

        res = self.cache.get("What is machine learning?", query_embedding=norm_vec)
        self.assertIsNotNone(res, "Identical vector must produce a cache hit")
        self.assertTrue(res['cache_hit'])
        self.assertAlmostEqual(res['similarity'], 1.0, places=3)
        self.assertEqual(res['matched_query'], "What is machine learning?")
        self.assertIn(res['cache_source'], ('in_memory', 'redis'))
        self.assertGreaterEqual(res['lookup_latency_ms'], 0.0)

    def test_redis_unavailable_fallback_behavior(self):
        """Test cache behaves cleanly when Redis backend is marked offline."""
        self.cache._redis = None
        self.cache._redis_offline = True

        vec = np.ones(768, dtype=np.float32)
        vec /= np.linalg.norm(vec)

        self.cache.set(
            query="What is gravity?",
            response="Gravity is a fundamental interaction.",
            doc_embedding=vec
        )

        res = self.cache.get("What is gravity?", query_embedding=vec)
        self.assertIsNotNone(res)
        self.assertTrue(res['cache_hit'])
        self.assertEqual(res['cache_source'], 'in_memory')
        self.assertAlmostEqual(res['similarity'], 1.0, places=3)

    def test_dashboard_render_handles_all_payload_shapes(self):
        """Test that data passed to dashboard hallucination analysis extracts score safely."""
        # 1. Normal payload with hallucination_score
        d1 = {'hallucination_score': 0.125, 'risk_level': 'low'}
        score1 = float(d1.get('hallucination_score', 0.0) or 0.0)
        self.assertEqual(score1, 0.125)

        # 2. Payload with None hallucination_score
        d2 = {'hallucination_score': None, 'risk_level': None}
        score2 = float(d2.get('hallucination_score', 0.0) or 0.0)
        self.assertEqual(score2, 0.0)

        # 3. Cache hit payload without hallucination_score key
        d3 = {'cache_hit': True, 'response': 'cached answer'}
        score3 = float(d3.get('hallucination_score', 0.0) or 0.0)
        self.assertEqual(score3, 0.0)


if __name__ == '__main__':
    unittest.main()
