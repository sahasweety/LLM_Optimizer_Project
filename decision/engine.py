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

    def _pareto_score(self, row) -> float:
        latency_norm = min(row['avg_latency_ms'] / 3000, 1.0)
        cost_norm    = min(row['total_cost_usd'] / 0.1, 1.0)
        halluc_norm  = row['avg_hallucination']
        return (latency_norm * 0.4) + (cost_norm * 0.3) + (halluc_norm * 0.3)

    def update_weights(self):
        db = self._get_db()
        if db is None:
            return
        try:
            with db.connect() as conn:
                rows = conn.execute(text('''
                    SELECT strategy,
                           AVG(avg_latency_ms)    AS avg_latency_ms,
                           SUM(total_cost_usd)    AS total_cost_usd,
                           AVG(avg_hallucination) AS avg_hallucination
                    FROM window_summaries
                    WHERE window_start > NOW() - INTERVAL '1 hour'
                    GROUP BY strategy
                ''')).fetchall()

            for row in rows:
                score = self._pareto_score(dict(row._mapping))
                self.weights[row.strategy] = 1.0 - score
        except Exception as e:
            logger.warning(f'DB query failed: {e}')

    def select_strategy(self) -> str:
        if random.random() < self.EPSILON:
            return random.choice(self.STRATEGIES)
        return max(self.weights, key=self.weights.get)

    def get_report(self) -> dict:
        return {
            'weights': self.weights,
            'recommended': self.select_strategy()
        }