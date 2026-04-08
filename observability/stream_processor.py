from kafka import KafkaConsumer
from collections import defaultdict
import json
import time
import logging

logger = logging.getLogger(__name__)

class StreamProcessor:
    WINDOW_SIZE = 60

    def __init__(self, db_writer):
        self._consumer = None
        self.db = db_writer
        self.window = defaultdict(list)
        self.window_start = time.time()

    def _get_consumer(self):
        """Lazy Kafka consumer – retries connection until available."""
        if self._consumer is None:
            try:
                self._consumer = KafkaConsumer(
                    'llm-events',
                    bootstrap_servers=['localhost:9092'],
                    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                    group_id='stream-processor',
                    auto_offset_reset='latest',
                    request_timeout_ms=10000,
                    session_timeout_ms=10000,
                )
                logger.info("Kafka consumer connected successfully.")
            except Exception as e:
                logger.warning(f"Kafka consumer connection failed: {e}")
                return None
        return self._consumer

    def _flush_window(self):
        if not self.window:
            return
        for strategy, events in self.window.items():
            summary = {
                'window_start': self.window_start,
                'strategy': strategy,
                'count': len(events),
                'avg_latency_ms': sum(e['latency_ms'] for e in events) / len(events),
                'avg_tokens': sum(e['tokens'] for e in events) / len(events),
                'total_cost_usd': sum(e['cost_usd'] for e in events),
                'avg_hallucination': sum(e['hallucination_score'] for e in events) / len(events)
            }
            self.db.write_window_summary(summary)
        self.window.clear()
        self.window_start = time.time()

    def run(self):
        while True:
            consumer = self._get_consumer()
            if consumer is None:
                logger.info("Waiting for Kafka to become available... retrying in 10s")
                time.sleep(10)
                continue
            try:
                for msg in consumer:
                    event = msg.value
                    strategy = event.get('strategy', 'unknown')
                    self.window[strategy].append(event)
                    if time.time() - self.window_start >= self.WINDOW_SIZE:
                        self._flush_window()
            except Exception as e:
                logger.warning(f"Kafka consumer error: {e} – reconnecting in 10s")
                self._consumer = None
                time.sleep(10)