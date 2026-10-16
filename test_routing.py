"""End-to-end multi-tier routing verification script."""
import requests

BASE = 'http://127.0.0.1:8081'

# Clear cache first so we force fresh LLM calls
try:
    cr = requests.delete(f'{BASE}/cache/clear', timeout=10)
    print(f'Cache cleared: {cr.json()}\n')
except Exception as e:
    print(f'Cache clear failed (continuing): {e}\n')

test_queries = [
    # Expected: fast (Llama via OpenRouter)
    ('fast',     'What is the boiling point of water?'),
    # Expected: balanced (Gemini)
    ('balanced', 'Write a creative short poem about autumn leaves'),
    # Expected: powerful (Claude via OpenRouter)
    ('powerful', 'Analyze and compare the advantages of supervised vs unsupervised learning with real-world examples'),
    # Expected: expert (GPT-4o-mini via OpenRouter)
    ('expert',   'Design a complete microservices architecture for an e-commerce platform with reasoning about scalability tradeoffs'),
]

print('=== MULTI-TIER ROUTING VERIFICATION ===\n')
all_ok = True
for expected_tier, q in test_queries:
    try:
        r = requests.post(f'{BASE}/query', json={'query': q}, timeout=60)
        if r.status_code != 200:
            print(f'HTTP {r.status_code} for: {q[:50]}\n{r.text[:300]}\n')
            all_ok = False
            continue
        d = r.json()
        status = 'FALLBACK' if d.get('fallback_occurred') else 'OK'
        actual_tier = d.get('tier', '?')
        tier_match = 'PASS' if actual_tier == expected_tier else 'MISMATCH'
        if actual_tier != expected_tier:
            all_ok = False
        complexity_val = d.get('complexity')
        complexity_str = f'{complexity_val:.3f}' if complexity_val is not None else 'N/A'
        print(f'Query    : {q[:65]}')
        print(f'Expected : {expected_tier:<10} | Got: {actual_tier:<10} | {tier_match}')
        print(f'Type     : {d.get("query_type","?")} | Complexity: {complexity_str}')
        print(f'Intended : {d.get("intended_provider","?")} / {d.get("intended_model","?")}')
        print(f'Actual   : {d.get("actual_provider","?")} / {d.get("actual_model","?")}')
        print(f'Status   : {status} | Latency: {d.get("latency_ms",0):.0f}ms | Cache: {d.get("cache_hit")}')
        if d.get('fallback_reason'):
            print(f'Fallback : {d["fallback_reason"]}')
        print()
    except Exception as e:
        print(f'ERROR for [{q[:50]}]: {e}\n')
        all_ok = False

print('=== RESULT:', 'ALL TIERS VERIFIED' if all_ok else 'SOME ISSUES DETECTED', '===')
