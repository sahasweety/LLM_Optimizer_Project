class PromptModule:
    TEMPLATES = {
        'qa': {
            'system': 'You are a precise assistant. Answer concisely and factually.',
            'prefix': 'Question: ',
            'suffix': '\nAnswer directly and cite sources if possible.'
        },
        'code': {
            'system': 'You are an expert programmer. Write clean, commented code.',
            'prefix': 'Task: ',
            'suffix': '\nProvide working code with explanation.'
        },
        'analysis': {
            'system': 'You are an analytical expert. Be structured and thorough.',
            'prefix': 'Analyze: ',
            'suffix': '\nStructure your response with clear sections.'
        },
        'summary': {
            'system': 'You are a summarization expert. Be concise.',
            'prefix': '',
            'suffix': '\nSummarize in 3-5 bullet points.'
        }
    }

    def detect_task_type(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ['code', 'function', 'write a', 'implement', 'debug']):
            return 'code'
        if any(w in q for w in ['analyze', 'compare', 'evaluate', 'assess']):
            return 'analysis'
        if any(w in q for w in ['summarize', 'summary', 'tldr', 'brief']):
            return 'summary'
        return 'qa'

    def optimize(self, query: str) -> dict:
        task = self.detect_task_type(query)
        template = self.TEMPLATES[task]
        return {
            'system': template['system'],
            'prompt': template['prefix'] + query + template['suffix'],
            'task_type': task
        }