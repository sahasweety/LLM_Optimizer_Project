import requests, json

# Test semantic cache: "What is ML?" should hit cache for "What is Machine Learning?"
r = requests.post("http://127.0.0.1:8081/query", json={"query": "What is ML?"}, timeout=30)
data = r.json()
print("Status:", r.status_code)
print("cache_hit:", data.get("cache_hit"))
print("similarity:", data.get("similarity"))
print("matched_query:", data.get("matched_query"))
print("cache_source:", data.get("cache_source"))
print("lookup_latency_ms:", data.get("lookup_latency_ms"))
print("response:", data.get("response", "")[:80])
print()

# Also test metrics
r2 = requests.get("http://127.0.0.1:8081/metrics", timeout=5)
print("Metrics:", json.dumps(r2.json(), indent=2))
