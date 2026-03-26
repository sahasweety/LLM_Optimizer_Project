class ModelSelector:
    MODELS = {
        'fast': {
            'name': 'llama-3.1-8b-instant',
            'provider': 'groq',
            'cost_per_1k': 0.0001,
            'max_tokens': 4096,
            'tier': 'fast'
        },
        'balanced': {
            'name': 'gemini-2.5-flash-lite',
            'provider': 'google',
            'cost_per_1k': 0.0002,
            'max_tokens': 4096,
            'tier': 'balanced'
        },
        'powerful': {
            'name': 'anthropic/claude-3-haiku',
            'provider': 'openrouter',
            'cost_per_1k': 0.00025,
            'max_tokens': 4096,
            'tier': 'powerful'
        },
        'expert': {
            'name': 'openai/gpt-4o-mini',
            'provider': 'openrouter',
            'cost_per_1k': 0.003,
            'max_tokens': 8192,
            'tier': 'expert'
        }
    }

    def _complexity_score(self, query: str) -> float:
        score = 0.0
        words = query.lower().split()
        score += min(len(query) / 300, 1.0) * 0.4
        complex_words = [
            'analyze', 'compare', 'evaluate', 'reasoning',
            'multi-step', 'explain', 'difference', 'implement',
            'how', 'why', 'what', 'describe', 'detail',
            'architecture', 'mechanism', 'algorithm', 'example'
        ]
        score += sum(1 for w in complex_words if w in words) * 0.08
        score += query.count('?') * 0.05
        score += min(len(words) / 20, 1.0) * 0.2
        return min(score, 1.0)

    def select(self, query: str) -> dict:
        score = self._complexity_score(query)
        if score < 0.18:
            tier = 'fast'
        elif score < 0.32:
            tier = 'balanced'
        elif score < 0.48:
            tier = 'powerful'
        else:
            tier = 'expert'
        return {'tier': tier, 'complexity': score, **self.MODELS[tier]}