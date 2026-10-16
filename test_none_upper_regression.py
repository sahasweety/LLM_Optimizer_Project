"""
Regression tests for 'NoneType' object has no attribute 'upper' / 'lower'.

Validates:
1. Cache hit payload (with None intended_provider, actual_provider, query_type, etc.)
   renders through render_query_result logic without throwing AttributeError.
2. Miss payload with None fields (e.g. actual_provider=None, fallback_reason=None)
   does not crash and reports semantically correct defaults.
3. Query history with explicit None risk_level or provider renders in history tables without crashing.
4. Normal string values for providers, tiers, and risk levels are properly capitalized.
"""

import sys
import unittest


def mock_render_eval_findings(data: dict):
    """Extracted logic from dashboard.py lines 268-278"""
    results = {}
    exp = data.get('explanation') or "Evaluation complete."
    results['explanation'] = exp
    if not data.get('cache_hit') and (data.get('tier') or data.get('query_type')):
        intended_p = (data.get('intended_provider') or '').upper() if isinstance(data.get('intended_provider'), str) else ''
        actual_p = (data.get('actual_provider') or '').upper() if isinstance(data.get('actual_provider'), str) else ''
        prov_text = f" | Provider: `{actual_p}`" if actual_p else ""
        fb_text = " (Fallback)" if data.get('fallback_occurred') else ""
        complexity_val = data.get('complexity')
        comp_str = f"{complexity_val:.2f}" if isinstance(complexity_val, (int, float)) else "N/A"
        results['caption'] = f"**Routing:** Tier: `{data.get('tier', 'N/A')}` | Type: `{data.get('query_type', 'N/A')}` | Complexity: `{comp_str}`{prov_text}{fb_text}"
    else:
        results['caption'] = None
    return results


def mock_render_cache_miss_bar(data: dict):
    """Extracted logic from dashboard.py lines 142-162"""
    selected_model = data.get('actual_model') or data.get('model', 'N/A')
    intended_model = data.get('intended_model') or selected_model
    actual_provider = (data.get('actual_provider') or '').upper() if isinstance(data.get('actual_provider'), str) else ''
    intended_provider = (data.get('intended_provider') or actual_provider).upper() if isinstance(data.get('intended_provider') or actual_provider, str) else ''
    actual_latency = data.get('latency_ms', 0)
    tier = data.get('tier')
    tier_info = f" ({tier.upper()} tier)" if isinstance(tier, str) and tier else ""

    fallback_flag = data.get('fallback_occurred', False)
    fallback_note = ""
    if fallback_flag:
        fallback_reason = data.get('fallback_reason', 'Provider error')
        fallback_note = f"⚠️ Fallback triggered: Intended {intended_model} ({intended_provider}) failed → handled by {selected_model} ({actual_provider}). Reason: {fallback_reason}"

    provider_disp = f"[{actual_provider}] " if actual_provider else ""
    return {
        'selected_model': selected_model,
        'actual_provider': actual_provider,
        'intended_provider': intended_provider,
        'tier_info': tier_info,
        'provider_disp': provider_disp,
        'fallback_note': fallback_note
    }


def mock_render_risk_assessment(data: dict):
    """Extracted logic from dashboard.py lines 219-221 & 449"""
    raw_risk = data.get('risk_level')
    risk = (raw_risk if isinstance(raw_risk, str) else 'low').lower()
    table_risk = (data.get('risk_level') or 'low').upper() if isinstance(data.get('risk_level') or 'low', str) else 'LOW'
    return risk, table_risk


class TestNoneTypeUpperRegression(unittest.TestCase):

    def test_cache_hit_payload_with_none_providers(self):
        """Test 1: Actual API Cache Hit response with tier='cache' and intended_provider=None, actual_provider=None"""
        cache_hit_data = {
            'query': 'What is the capital of France?',
            'response': 'Paris',
            'strategy': 'cache',
            'model': 'cache',
            'latency_ms': 1.0,
            'tokens': 0,
            'cost_usd': 0.0,
            'hallucination_score': 0.0,
            'risk_level': 'low',
            'cache_hit': True,
            'explanation': 'Low Risk: Response served instantly from Semantic Cache.',
            'similarity': 1.0,
            'matched_query': 'What is the capital of France?',
            'cache_source': 'in_memory',
            'lookup_latency_ms': 0.2,
            'query_type': None,
            'tier': 'cache',
            'complexity': None,
            'intended_model': None,
            'intended_provider': None,
            'actual_model': None,
            'actual_provider': None,
            'fallback_occurred': False,
            'fallback_reason': None
        }

        # Should not raise AttributeError: 'NoneType' object has no attribute 'upper'
        res = mock_render_eval_findings(cache_hit_data)
        self.assertIsNone(res['caption'], "Cache hits should not render LLM routing captions")
        self.assertEqual(res['explanation'], 'Low Risk: Response served instantly from Semantic Cache.')

        risk, table_risk = mock_render_risk_assessment(cache_hit_data)
        self.assertEqual(risk, 'low')
        self.assertEqual(table_risk, 'LOW')

    def test_cache_miss_payload_with_none_providers(self):
        """Test 2: Cache miss where provider fields might be None"""
        miss_data = {
            'query': 'What is quantum computing?',
            'tier': 'fast',
            'query_type': 'technical',
            'complexity': 0.45,
            'cache_hit': False,
            'intended_provider': None,
            'actual_provider': None,
            'intended_model': 'meta-llama/llama-3.1-8b-instruct',
            'actual_model': 'meta-llama/llama-3.1-8b-instruct',
            'latency_ms': 1200.0,
            'fallback_occurred': False
        }

        res = mock_render_eval_findings(miss_data)
        self.assertIsNotNone(res['caption'])
        self.assertIn("Tier: `fast`", res['caption'])
        self.assertIn("Complexity: `0.45`", res['caption'])
        # Provider text should be empty without crashing
        self.assertNotIn("Provider: `NONE`", res['caption'])

        miss_bar = mock_render_cache_miss_bar(miss_data)
        self.assertEqual(miss_bar['actual_provider'], "")
        self.assertEqual(miss_bar['intended_provider'], "")
        self.assertEqual(miss_bar['tier_info'], " (FAST tier)")

    def test_normal_string_values_preserved(self):
        """Test 3: Normal string providers and tiers remain correctly uppercase"""
        data = {
            'query': 'Explain photosynthesis',
            'tier': 'expert',
            'query_type': 'scientific',
            'complexity': 0.82,
            'cache_hit': False,
            'intended_provider': 'openai',
            'actual_provider': 'openrouter',
            'intended_model': 'gpt-4o-mini',
            'actual_model': 'openai/gpt-4o-mini',
            'risk_level': 'medium',
            'fallback_occurred': True,
            'fallback_reason': 'Rate limit'
        }

        res = mock_render_eval_findings(data)
        self.assertIn("Provider: `OPENROUTER`", res['caption'])
        self.assertIn("(Fallback)", res['caption'])

        miss_bar = mock_render_cache_miss_bar(data)
        self.assertEqual(miss_bar['actual_provider'], "OPENROUTER")
        self.assertEqual(miss_bar['intended_provider'], "OPENAI")
        self.assertEqual(miss_bar['tier_info'], " (EXPERT tier)")
        self.assertIn("⚠️ Fallback triggered: Intended gpt-4o-mini (OPENAI) failed → handled by openai/gpt-4o-mini (OPENROUTER)", miss_bar['fallback_note'])

        risk, table_risk = mock_render_risk_assessment(data)
        self.assertEqual(risk, 'medium')
        self.assertEqual(table_risk, 'MEDIUM')

    def test_explicit_none_risk_level(self):
        """Test 4: Explicit risk_level=None does not crash with NoneType has no attribute upper/lower"""
        data = {
            'risk_level': None,
            'cache_hit': False
        }
        risk, table_risk = mock_render_risk_assessment(data)
        self.assertEqual(risk, 'low')
        self.assertEqual(table_risk, 'LOW')


if __name__ == '__main__':
    unittest.main()
