class PromptModule:
    TEMPLATES = {
        'coding': {
            'system': 'You are an expert programmer. Write clean, optimal, well-commented, and secure code. Follow best practices for the language specified.',
            'prefix': 'Coding Task: ',
            'suffix': '\nProvide the complete, compiling solution with clean structure, explanations of complexity (time/space), and edge-case handling.'
        },
        'reasoning': {
            'system': 'You are a logical reasoning expert. Solve this problem step-by-step using first principles and verify each logical connection.',
            'prefix': 'Reasoning Problem: ',
            'suffix': '\nBreak down your thinking process step-by-step. Show all intermediate work and verify your final conclusion for logical consistency.'
        },
        'math': {
            'system': 'You are a mathematician. Solve this mathematical problem step-by-step. State any assumptions and prove or calculate the result clearly.',
            'prefix': 'Mathematical Problem: ',
            'suffix': '\nShow the exact mathematical derivation or calculation. Format equations cleanly and provide the final answer clearly labeled.'
        },
        'creative': {
            'system': 'You are a creative writer. Craft a highly engaging, expressive, and original piece based on the prompt. Focus on vivid imagery, tone, and flow.',
            'prefix': 'Creative Request: ',
            'suffix': '\nWrite creatively and expressively, ensuring appropriate tone, pacing, and style.'
        },
        'summarization': {
            'system': 'You are a summarization expert. Extract the core arguments, facts, and conclusions from the text. Be concise and retain maximum key information.',
            'prefix': 'Summarize the following text: ',
            'suffix': '\nProvide a concise 3-5 bullet point summary, followed by a one-sentence TL;DR.'
        },
        'translation': {
            'system': 'You are a professional translator. Translate the text accurately while preserving the original tone, context, idioms, and nuances.',
            'prefix': 'Translate: ',
            'suffix': '\nProvide the accurate translation. If there are multiple cultural interpretations or nuances, explain them briefly.'
        },
        'analysis': {
            'system': 'You are an expert analyst. Provide a structured, thorough, objective, and data-driven analysis of the topic. Cover pros, cons, and implications.',
            'prefix': 'Analyze: ',
            'suffix': '\nStructure your response with clear headings: Background, Detailed Analysis, Key Takeaways, and Conclusion.'
        },
        'factual': {
            'system': 'You are a precise, objective factual assistant. Provide accurate, verified information. If details are unverified or unknown, state so clearly.',
            'prefix': 'Factual Inquiry: ',
            'suffix': '\nProvide direct, factual answers. Cite reliable sources or historical context where applicable.'
        },
        'current_events': {
            'system': 'You are a current events analyst. Explain the context, main actors, and latest developments. Acknowledge any lack of real-time info if applicable.',
            'prefix': 'Current Event Question: ',
            'suffix': '\nProvide the context and details of these events. Clearly state the timeframe of your information.'
        },
        'sensitive_info': {
            'system': 'You are a security and privacy guardian. Carefully evaluate safety, privacy, and confidentiality constraints. Do not disclose private, personal, or confidential data.',
            'prefix': 'Sensitive Query: ',
            'suffix': '\nEnsure the response is safe, respects privacy, does not disclose private facts, and adheres to ethical guidelines.'
        },
        'general_chat': {
            'system': 'You are a warm, helpful, and friendly conversational assistant. Keep the conversation natural, helpful, and polite.',
            'prefix': '',
            'suffix': ''
        }
    }

    def detect_task_type(self, query: str) -> str:
        q = query.lower()
        
        # Coding
        coding_keywords = ['code', 'function', 'class', 'write a program', 'script', 'python', 'javascript', 'java', 'html', 'css', 'sql', 'bug', 'debug', 'exception']
        if any(w in q for w in coding_keywords):
            return 'coding'
            
        # Math
        math_keywords = ['math', 'calculate', 'solve for', 'equation', 'plus', 'minus', 'multiply', 'divide', 'derivative', 'integral', 'fraction', 'algebra', 'geometry']
        if any(w in q for w in math_keywords):
            return 'math'
            
        # Reasoning
        reasoning_keywords = ['riddle', 'puzzle', 'logic', 'reason', 'proof', 'step-by-step', 'deduce', 'contradiction']
        if any(w in q for w in reasoning_keywords):
            return 'reasoning'
            
        # Translation
        translation_keywords = ['translate', 'translation', 'in spanish', 'in french', 'in german', 'in chinese', 'in japanese', 'how to say']
        if any(w in q for w in translation_keywords):
            return 'translation'
            
        # Summarization
        summary_keywords = ['summarize', 'summary', 'tldr', 'brief', 'gist', 'outline']
        if any(w in q for w in summary_keywords):
            return 'summarization'
            
        # Analysis
        analysis_keywords = ['analyze', 'compare', 'evaluate', 'assess', 'analysis', 'pros and cons', 'implications']
        if any(w in q for w in analysis_keywords):
            return 'analysis'
            
        # Creative
        creative_keywords = ['poem', 'story', 'creative', 'brainstorm', 'ideas', 'essay', 'compose', 'write a letter', 'draft']
        if any(w in q for w in creative_keywords):
            return 'creative'
            
        # Sensitive/Private
        sensitive_keywords = ['private', 'confidential', 'secret', 'salary of', 'net worth of', 'password', 'personal detail', 'private information']
        if any(w in q for w in sensitive_keywords):
            return 'sensitive_info'
            
        # Current Events
        current_keywords = ['news', 'current event', 'latest on', 'what happened today', 'recent developments', 'election 2026', 'weather today']
        if any(w in q for w in current_keywords):
            return 'current_events'
            
        # General chat
        chat_keywords = ['hello', 'hi ', 'hey ', 'how are you', 'tell me a joke', 'chat with me']
        if any(w in q for w in chat_keywords):
            return 'general_chat'
            
        return 'factual'

    def optimize(self, query: str) -> dict:
        task = self.detect_task_type(query)
        template = self.TEMPLATES[task]
        return {
            'system': template['system'],
            'prompt': template['prefix'] + query + template['suffix'],
            'task_type': task
        }