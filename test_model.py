from optimization.model_selector import ModelSelector

m = ModelSelector()

queries = [
    'What is AI?',
    'How does deep learning work?',
    'Explain transformer architecture in detail',
    'Analyze and compare supervised vs unsupervised learning with real world examples and evaluate',
    'Analyze and compare the differences between supervised learning unsupervised learning and reinforcement learning with real world examples evaluate which approach works best and explain why transformers replaced RNNs'
]

for q in queries:
    r = m.select(q)
    print(f"Score: {r['complexity']:.3f} | Tier: {r['tier']:10} | Model: {r['name']:35} | Query: {q[:55]}")