from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

import time

class DBWriter:
    def __init__(self):
        self._engine = None
        self._db_offline = False
        self._last_db_check = 0

    def _get_engine(self):
        """Lazy DB engine – connects only when needed with a 60-second cool-off retry."""
        now = time.time()
        if self._db_offline and (now - self._last_db_check < 60):
            return None

        if self._engine is None or self._db_offline:
            self._last_db_check = now
            db_url = os.getenv('DATABASE_URL')
            if not db_url:
                logger.warning("DATABASE_URL not set – DBWriter disabled.")
                self._db_offline = True
                return None
            try:
                self._engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self._db_offline = False
                logger.info("DBWriter connected to database.")
            except Exception as e:
                logger.warning(f"DBWriter DB connection failed: {e}. Retrying in 60s.")
                self._engine = None
                self._db_offline = True
                return None
        return self._engine

    def write_event(self, event: dict):
        engine = self._get_engine()
        if engine is None:
            return
        try:
            with engine.connect() as conn:
                conn.execute(text('''
                    INSERT INTO llm_events 
                    (event_id, query, strategy, model, latency_ms, 
                     tokens, cost_usd, hallucination_score, risk_level)
                    VALUES
                    (:event_id, :query, :strategy, :model, :latency_ms,
                     :tokens, :cost_usd, :hallucination_score, :risk_level)
                '''), event)
                conn.commit()
        except Exception as e:
            logger.warning(f"DBWriter write_event failed: {e}")
            self._engine = None

    def write_window_summary(self, summary: dict):
        engine = self._get_engine()
        if engine is None:
            return
        try:
            with engine.connect() as conn:
                conn.execute(text('''
                    INSERT INTO window_summaries
                    (window_start, strategy, event_count, avg_latency_ms,
                     avg_tokens, total_cost_usd, avg_hallucination)
                    VALUES
                    (:window_start, :strategy, :count, :avg_latency_ms,
                     :avg_tokens, :total_cost_usd, :avg_hallucination)
                '''), summary)
                conn.commit()
        except Exception as e:
            logger.warning(f"DBWriter write_window_summary failed: {e}")
            self._engine = None