from kafka import KafkaProducer
import json
import time
import uuid

class FeedbackCollector:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        self.topic = 'llm-events'
        self.dlq = 'llm-events-dlq'

    def emit(self, event: dict):
        payload = {
            'event_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            **event
        }
        try:
            future = self.producer.send(self.topic, payload)
            future.get(timeout=5)
        except Exception as e:
            self.producer.send(self.dlq, {**payload, 'error': str(e)})

    def emit_llm_call(self, query, response, strategy,
                      model, latency_ms, tokens, cost,
                      hallucination_score):
        self.emit({
            'type': 'llm_call',
            'query': query[:200],
            'response_preview': response[:100],
            'strategy': strategy,
            'model': model,
            'latency_ms': latency_ms,
            'tokens': tokens,
            'cost_usd': cost,
            'hallucination_score': hallucination_score
        })