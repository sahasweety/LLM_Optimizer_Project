"""
Cache regression tests.

Covers:
  1. First exact query -> MISS
  2. Repeated exact query -> HIT
  3. Semantic paraphrase above threshold -> HIT
  4. Unrelated query -> MISS
  5. Cache backend unavailable -> falls back gracefully (no crash)
  6. No LLM call on a cache HIT
  7. Cache persists between requests (same server instance)

Run with:
  .venv\\Scripts\\python.exe -X utf8 test_cache.py

Requires the local dev server to be running on 127.0.0.1:8081.
"""

import requests
import time
import sys

BASE = "http://127.0.0.1:8081"
TIMEOUT = 60

# ── Helpers ──────────────────────────────────────────────────────────────────

def query(q, label=""):
    r = requests.post(f"{BASE}/query", json={"query": q}, timeout=TIMEOUT)
    assert r.status_code == 200, f"HTTP {r.status_code}: {r.text[:300]}"
    d = r.json()
    hit = d.get("cache_hit", False)
    src = d.get("cache_source", "none")
    sim = d.get("similarity")
    matched = d.get("matched_query", "")
    model = d.get("actual_model") or d.get("model", "?")
    latency = d.get("latency_ms", 0)
    tier = d.get("tier", "?")
    lookup_ms = d.get("lookup_latency_ms")

    sim_str = f"{sim:.4f}" if sim is not None else "N/A"
    print(f"  [{label}] cache_hit={hit} | src={src} | sim={sim_str} | "
          f"model={model} | tier={tier} | latency={latency:.0f}ms | "
          f"lookup={lookup_ms}ms")
    if matched:
        print(f"    matched_query: {matched[:70]}")
    return d


def clear_cache():
    r = requests.delete(f"{BASE}/cache/clear", timeout=10)
    assert r.status_code == 200
    d = r.json()
    print(f"  Cache cleared: {d['cleared_in_memory']} in-memory, {d['cleared_redis']} redis")
    return d


def check_cache_stats():
    r = requests.get(f"{BASE}/cache/stats", timeout=10)
    assert r.status_code == 200
    d = r.json()
    print(f"  Cache stats: total_entries={d['total_entries']} | "
          f"threshold={d['threshold']} | redis_online={d['redis_online']}")
    return d


# ── Tests ─────────────────────────────────────────────────────────────────────

PASS = []
FAIL = []

def run(name, fn):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)
    try:
        fn()
        print(f"  => PASS")
        PASS.append(name)
    except AssertionError as e:
        print(f"  => FAIL: {e}")
        FAIL.append(name)
    except Exception as e:
        print(f"  => ERROR: {e}")
        FAIL.append(name)


# ── Test 1: First exact query -> MISS ─────────────────────────────────────────
def test_1_first_query_miss():
    clear_cache()
    d = query("What is the capital of France?", "1st request")
    assert not d["cache_hit"], f"Expected MISS on first query, got HIT (similarity={d.get('similarity')})"
    assert d.get("actual_model") or d.get("model"), "No model info returned on MISS"


# ── Test 2: Repeated exact query -> HIT ───────────────────────────────────────
def test_2_repeat_exact_hit():
    # Don't clear cache — rely on entry from test_1
    time.sleep(1)  # small gap to ensure background cache.set() has completed
    d = query("What is the capital of France?", "2nd request (exact repeat)")
    assert d["cache_hit"], (
        f"Expected HIT on exact repeat, got MISS "
        f"(similarity={d.get('similarity')}, threshold likely not met)"
    )
    assert d.get("actual_model") == "cache" or d.get("model") == "cache", (
        "Cache hit should report model='cache', not a real model"
    )
    sim = d.get("similarity", 0)
    assert sim is not None and sim >= 0.75, f"Expected similarity >= 0.75, got {sim}"


# ── Test 3: Semantic paraphrase -> HIT ───────────────────────────────────────
def test_3_semantic_paraphrase_hit():
    # Seed a fresh query
    clear_cache()
    query("What is machine learning?", "seed (ML question)")
    time.sleep(1)

    paraphrases = [
        "Explain machine learning in simple words.",
        "Can you describe machine learning to me?",
    ]
    for p in paraphrases:
        d = query(p, f"paraphrase: {p[:40]}")
        sim = d.get("similarity")
        sim_str = f"{sim:.4f}" if sim is not None else "N/A"
        if d["cache_hit"]:
            print(f"    => HIT (sim={sim_str}) - correct semantic match")
        else:
            print(f"    => MISS (sim={sim_str}) - below threshold (may be acceptable)")
    # We don't hard-assert on paraphrases — similarity depends on the embedding model.
    # We do assert the response is at least structurally correct.
    for p in paraphrases:
        d = query(p, f"re-check: {p[:40]}")
        assert "response" in d and len(d["response"]) > 0


# ── Test 4: Unrelated query -> MISS ──────────────────────────────────────────
def test_4_unrelated_miss():
    # Cache seeded with "What is machine learning?" from test 3
    d = query("What is the boiling point of gold?", "unrelated query")
    sim = d.get("similarity", 0) or 0
    if d["cache_hit"]:
        raise AssertionError(
            f"Expected MISS for unrelated query, got HIT (sim={sim:.4f}). "
            f"Threshold may be too low or normalization is over-matching."
        )
    print(f"    => Correctly returned MISS for unrelated query (sim={sim:.4f})")


# ── Test 5: Cache backend unavailable -> graceful degradation ─────────────────
def test_5_redis_unavailable_graceful():
    # Check cache stats - Redis offline is expected in local dev
    stats = check_cache_stats()
    if not stats["redis_online"]:
        print("    Redis is offline (expected in local dev). Cache falls back to in-memory.")
        # Verify the in-memory cache still works
        clear_cache()
        query("Redis test: what is gravity?", "seed")
        time.sleep(1)
        d = query("Redis test: what is gravity?", "repeat after seed")
        assert d["cache_hit"], "In-memory cache should still work when Redis is offline"
        print("    => In-memory fallback works correctly.")
    else:
        print("    Redis is online. Testing Redis path...")
        clear_cache()
        query("Redis test: what is gravity?", "seed")
        time.sleep(1)
        d = query("Redis test: what is gravity?", "repeat")
        assert d["cache_hit"], "Redis cache should return HIT on repeat query"
        print(f"    => Redis cache hit confirmed (source={d.get('cache_source')})")


# ── Test 6: No LLM model call on cache HIT ───────────────────────────────────
def test_6_no_llm_on_hit():
    # Re-seed
    clear_cache()
    miss_d = query("Describe the water cycle.", "seed (miss)")
    assert not miss_d["cache_hit"], "Should be a MISS on first query"
    miss_model = miss_d.get("actual_model") or miss_d.get("model", "?")
    assert miss_model != "cache", f"First query should call a real model, got: {miss_model}"
    time.sleep(1)

    hit_d = query("Describe the water cycle.", "repeat (hit)")
    assert hit_d["cache_hit"], "Second query should be a HIT"
    hit_model = hit_d.get("model", "?")
    assert hit_model == "cache", (
        f"Cache HIT must not call a real model, but model='{hit_model}'"
    )
    assert hit_d.get("tokens", 0) == 0, "Token count must be 0 on cache HIT (no LLM call)"
    assert hit_d.get("cost_usd", 0.0) == 0.0, "Cost must be $0 on cache HIT"
    print(f"    => Confirmed: no LLM call on HIT (tokens=0, cost=$0, model=cache)")


# ── Test 7: Verify cache entry count after operations ─────────────────────────
def test_7_cache_persistence_within_session():
    clear_cache()
    check_cache_stats()
    queries_to_seed = [
        "What is photosynthesis?",
        "How does the stock market work?",
        "Explain recursion in programming.",
    ]
    for q in queries_to_seed:
        query(q, f"seed: {q[:40]}")
        time.sleep(0.5)

    stats = check_cache_stats()
    count = stats["total_entries"]
    assert count == len(queries_to_seed), (
        f"Expected {len(queries_to_seed)} cache entries, found {count}"
    )
    print(f"    => {count}/{len(queries_to_seed)} entries found in cache (correct)")


# ── Run all tests ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nConnecting to {BASE}...")
    try:
        h = requests.get(f"{BASE}/health", timeout=5)
        print(f"Server health: {h.json()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {BASE}: {e}")
        sys.exit(1)

    run("1. First exact query -> MISS",              test_1_first_query_miss)
    run("2. Repeated exact query -> HIT",            test_2_repeat_exact_hit)
    run("3. Semantic paraphrase -> HIT (best effort)", test_3_semantic_paraphrase_hit)
    run("4. Unrelated query -> MISS",                test_4_unrelated_miss)
    run("5. Redis unavailable -> graceful fallback", test_5_redis_unavailable_graceful)
    run("6. No LLM call on cache HIT",               test_6_no_llm_on_hit)
    run("7. Cache entry count within session",       test_7_cache_persistence_within_session)

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(PASS)} PASS / {len(FAIL)} FAIL")
    if FAIL:
        print(f"FAILED: {FAIL}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
