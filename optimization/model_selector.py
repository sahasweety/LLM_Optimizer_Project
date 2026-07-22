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

    def _detect_query_type(self, query: str) -> str:
        q = query.lower()
        
        # Coding detection
        coding_keywords = [
            'code', 'function', 'class', 'write a program', 'script', 'python', 
            'javascript', 'java', 'html', 'css', 'sql', 'bug', 'debug', 'exception', 
            'compile', 'runtime', 'api endpoint', 'json', 'yaml', 'regex'
        ]
        if any(w in q for w in coding_keywords):
            return 'coding'
            
        # Reasoning detection
        reasoning_keywords = [
            'math', 'calculate', 'riddle', 'puzzle', 'solve for', 'equation', 
            'logic', 'reason', 'proof', 'step-by-step', 'algorithm', 'complexity', 
            'optimization', 'derive', 'differentiate'
        ]
        if any(w in q for w in reasoning_keywords):
            return 'reasoning'
            
        # Creative detection
        creative_keywords = [
            'poem', 'story', 'creative', 'brainstorm', 'ideas', 'essay', 'compose', 
            'write a letter', 'email template', 'congratulate', 'fictional', 
            'roleplay', 'draft'
        ]
        if any(w in q for w in creative_keywords):
            return 'creative'
            
        # Default/Factual
        return 'factual'

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
        q_type = self._detect_query_type(query)
        
        # Intent-driven routing matrix
        if q_type == 'coding':
            # Coding requires high logical coding capability: route to powerful or expert
            tier = 'expert' if score >= 0.3 else 'powerful'
        elif q_type == 'reasoning':
            # Reasoning requires step-by-step logic: route to expert or powerful/balanced
            if score >= 0.4:
                tier = 'expert'
            elif score >= 0.2:
                tier = 'powerful'
            else:
                tier = 'balanced'
        elif q_type == 'creative':
            # Creative needs flexibility and context: balanced or fast
            tier = 'balanced' if score >= 0.3 else 'fast'
        else: # factual
            # Factual is simple info retrieval: fast or balanced or powerful (if very complex analysis needed)
            if score >= 0.5:
                tier = 'powerful'
            elif score >= 0.2:
                tier = 'balanced'
            else:
                tier = 'fast'
                
        return {'tier': tier, 'complexity': score, 'query_type': q_type, **self.MODELS[tier]}