from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

class DBWriter:
    def __init__(self):
        self.engine = create_engine(os.getenv('DATABASE_URL'))

    def write_event(self, event: dict):
        with self.engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO llm_events 
                (event_id, query, strategy, model, latency_ms, 
                 tokens, cost_usd, hallucination_score, risk_level)
                VALUES
                (:event_id, :query, :strategy, :model, :latency_ms,
                 :tokens, :cost_usd, :hallucination_score, :risk_level)
            '''), event)
            conn.commit()

    def write_window_summary(self, summary: dict):
        with self.engine.connect() as conn:
            conn.execute(text('''
                INSERT INTO window_summaries
                (window_start, strategy, event_count, avg_latency_ms,
                 avg_tokens, total_cost_usd, avg_hallucination)
                VALUES
                (:window_start, :strategy, :count, :avg_latency_ms,
                 :avg_tokens, :total_cost_usd, :avg_hallucination)
            '''), summary)
            conn.commit()