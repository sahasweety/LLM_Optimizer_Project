import random
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class DecisionEngine:
    STRATEGIES = ['cache', 'prompt+model', 'baseline']
    EPSILON = 0.1

    def __init__(self):
        self._db = None
        self.weights = {s: 1.0 for s in self.STRATEGIES}
        self.in_memory_history = []  # Fallback history storage: list of dicts

    def _get_db(self):
        """Lazy DB engine – returns None if DATABASE_URL is missing or unreachable."""
        if self._db is None:
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                logger.warning("DATABASE_URL not set – DecisionEngine running without DB.")
                return None
            try:
                self._db = create_engine(db_url, pool_pre_ping=True)
                # Quick connectivity check
                with self._db.connect() as conn:
                    conn.execute(text("SELECT 1"))
                logger.info("DecisionEngine DB connected successfully.")
            except Exception as e:
                logger.warning(f"DecisionEngine DB connection failed: {e}")
                self._db = None
                return None
        return self._db

    def _calculate_score(self, avg_latency, avg_cost, avg_tokens, avg_hallucination, cache_hit_rate, count) -> float:
        # Normalize each metric (0.0 to 1.0 scale)
        latency_norm = min(avg_latency / 3000.0, 1.0)
        cost_norm    = min(avg_cost / 0.05, 1.0)
        tokens_norm  = min(avg_tokens / 2000.0, 1.0)
        halluc_norm  = min(avg_hallucination, 1.0)
        
        # Cache hit rate is positive, so penalty is (1.0 - cache_hit_rate)
        cache_penalty = 1.0 - cache_hit_rate
        
        # Historical performance reliability penalty:
        # Strategies with very few runs are penalized slightly to encourage exploration,
        # but once established, we want to favor strategies that perform reliably.
        exploration_penalty = max(0.0, 1.0 - (count / 10.0))
        
        # Weighted sum of penalties based on production priorities:
        # Latency (25%), Cost (20%), Tokens (15%), Hallucination (20%), Cache Hit (15%), Exploration (5%)
        total_penalty = (
            (latency_norm * 0.25) +
            (cost_norm * 0.20) +
            (tokens_norm * 0.15) +
            (halluc_norm * 0.20) +
            (cache_penalty * 0.15) +
            (exploration_penalty * 0.05)
        )
        return total_penalty

    def log_query(self, strategy, latency_ms, tokens, cost_usd, hallucination_score, cache_hit):
        """Append query metrics to the in-memory log and trigger weights update."""
        self.in_memory_history.append({
            'strategy': strategy,
            'latency_ms': latency_ms,
            'tokens': tokens,
            'cost_usd': cost_usd,
            'hallucination_score': hallucination_score,
            'cache_hit': cache_hit
        })
        # Bound in-memory history to last 100 queries to prevent memory leaks
        if len(self.in_memory_history) > 100:
            self.in_memory_history.pop(0)
        self.update_weights()

    def update_weights(self):
        db = self._get_db()
        stats = {s: [] for s in self.STRATEGIES}
        
        # 1. Fetch recent events from database if available
        if db is not None:
            try:
                with db.connect() as conn:
                    rows = conn.execute(text('''
                        SELECT strategy, latency_ms, tokens, cost_usd, hallucination_score
                        FROM llm_events
                        WHERE timestamp > EXTRACT(EPOCH FROM NOW()) - 3600
                    ''')).fetchall()
                    
                    for row in rows:
                        strat = row.strategy
                        if strat == 'model_selection':
                            strat = 'baseline'
                        if strat in stats:
                            stats[strat].append({
                                'latency_ms': row.latency_ms,
                                'tokens': row.tokens,
                                'cost_usd': row.cost_usd,
                                'hallucination_score': row.hallucination_score,
                                'cache_hit': (row.strategy == 'cache')
                            })
            except Exception as e:
                logger.warning(f"Failed to fetch DB stats: {e}")

        # 2. Add/fallback to in-memory history logs
        for item in self.in_memory_history:
            strat = item['strategy']
            if strat == 'model_selection':
                strat = 'baseline'
            if strat in stats:
                stats[strat].append(item)

        # 3. Calculate weights based on collected stats
        for strat in self.STRATEGIES:
            items = stats[strat]
            if not items:
                # Default weight if no history is present for this strategy
                self.weights[strat] = 1.0
                continue
                
            avg_latency = sum(x['latency_ms'] for x in items) / len(items)
            avg_cost = sum(x['cost_usd'] for x in items) / len(items)
            avg_tokens = sum(x['tokens'] for x in items) / len(items)
            avg_halluc = sum(x['hallucination_score'] for x in items) / len(items)
            cache_hits = sum(1 for x in items if x.get('cache_hit', False))
            cache_hit_rate = cache_hits / len(items)
            count = len(items)
            
            penalty = self._calculate_score(avg_latency, avg_cost, avg_tokens, avg_halluc, cache_hit_rate, count)
            self.weights[strat] = max(0.0, 1.0 - penalty)

    def select_strategy(self) -> str:
        if random.random() < self.EPSILON:
            return random.choice(self.STRATEGIES)
        return max(self.weights, key=self.weights.get)

    def get_report(self) -> dict:
        return {
            'weights': self.weights,
            'recommended': self.select_strategy()
        }