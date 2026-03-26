from kafka import KafkaConsumer
from collections import defaultdict
import json
import time

class StreamProcessor:
    WINDOW_SIZE = 60

    def __init__(self, db_writer):
        self.consumer = KafkaConsumer(
            'llm-events',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='stream-processor',
            auto_offset_reset='latest'
        )
        self.db = db_writer
        self.window = defaultdict(list)
        self.window_start = time.time()

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
        for msg in self.consumer:
            event = msg.value
            strategy = event.get('strategy', 'unknown')
            self.window[strategy].append(event)
            if time.time() - self.window_start >= self.WINDOW_SIZE:
                self._flush_window()