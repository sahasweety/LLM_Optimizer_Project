from .cache_module import CacheModule
from .prompt_module import PromptModule
from .model_selector import ModelSelector
import time

class OptimizationController:
    def __init__(self):
        self.cache = CacheModule(threshold=0.65)
        self.prompt = PromptModule()
        self.selector = ModelSelector()

    def process(self, query: str) -> dict:
        start = time.time()

        # Step 1: Check cache first
        cached = self.cache.get(query)
        if cached:
            return {
                'response': cached['response'],
                'strategy': 'cache',
                'latency_ms': (time.time() - start) * 1000,
                'cache_hit': True,
                'similarity': cached['similarity']
            }

        # Step 2: Select model based on complexity
        model_info = self.selector.select(query)
        complexity = model_info['complexity']

        # Step 3: Decide strategy based on complexity
        # Simple query (complexity < 0.3) → just use fast model, no prompt optimization
        # Complex query (complexity >= 0.3) → use prompt optimization + model selection
        if complexity < 0.3:
            strategy = 'model_selection'
            prompt = query
            system = 'You are a helpful assistant. Answer clearly and concisely.'
        else:
            strategy = 'prompt+model'
            optimized = self.prompt.optimize(query)
            prompt = optimized['prompt']
            system = optimized['system']

        return {
            'system': system,
            'prompt': prompt,
            'task_type': model_info.get('tier', 'fast'),
            'model': model_info,
            'strategy': strategy,
            'latency_ms': (time.time() - start) * 1000,
            'cache_hit': False
        }