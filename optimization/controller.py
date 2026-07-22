from .cache_module import CacheModule
from .prompt_module import PromptModule
from .model_selector import ModelSelector
import time

class OptimizationController:
    def __init__(self):
        self.cache = CacheModule()
        self.prompt = PromptModule()
        self.selector = ModelSelector()

    def process(self, query: str, engine=None) -> dict:
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

        # Step 2: Select model based on complexity and intent
        model_info = self.selector.select(query)
        complexity = model_info['complexity']

        # Step 3: Decide strategy based on DecisionEngine (if provided) or fallback complexity
        if engine is not None:
            rec_strategy = engine.select_strategy()
            if rec_strategy == 'baseline':
                strategy = 'model_selection'
            elif rec_strategy == 'cache':
                # If DecisionEngine recommends cache but we had a cache miss, we fall back to model selection
                strategy = 'model_selection'
            else:
                strategy = rec_strategy
        else:
            strategy = 'model_selection' if complexity < 0.3 else 'prompt+model'

        # Step 4: Apply prompt optimization if strategy calls for it
        if strategy == 'prompt+model':
            optimized = self.prompt.optimize(query)
            prompt = optimized['prompt']
            system = optimized['system']
        else:
            prompt = query
            system = 'You are a helpful assistant. Answer clearly and concisely.'

        return {
            'system': system,
            'prompt': prompt,
            'task_type': model_info.get('tier', 'fast'),
            'model': model_info,
            'strategy': strategy,
            'latency_ms': (time.time() - start) * 1000,
            'cache_hit': False
        }