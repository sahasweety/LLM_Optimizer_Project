from .prompt_module import PromptModule

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
        
        # Centralized type classification via PromptModule
        prompt_mod = PromptModule()
        q_type = prompt_mod.detect_task_type(query)
        
        # Comprehensive intent-driven routing matrix
        if q_type in ('coding', 'math', 'reasoning', 'sensitive_info'):
            # Demands advanced logical syntax or privacy guards: powerful/expert
            tier = 'expert' if score >= 0.3 else 'powerful'
        elif q_type in ('summarization', 'translation', 'analysis'):
            # Needs structured text translation/summaries: balanced/powerful
            tier = 'powerful' if score >= 0.4 else 'balanced'
        elif q_type in ('creative', 'current_events'):
            # Dynamic creative writing or temporal updates: balanced/fast
            tier = 'balanced' if score >= 0.3 else 'fast'
        else: # general_chat, factual QA
            # Fast response / simple chat: fast/balanced
            tier = 'balanced' if score >= 0.4 else 'fast'
                
        return {'tier': tier, 'complexity': score, 'query_type': q_type, **self.MODELS[tier]}