import json
import time
import uuid
import logging

logger = logging.getLogger(__name__)

class FeedbackCollector:
    def __init__(self):
        self._producer = None
        self._kafka_offline = False
        self._last_kafka_check = 0
        self.topic = 'llm-events'
        self.dlq = 'llm-events-dlq'

    def _get_producer(self):
        """Lazy Kafka producer – connect only when first needed with 60-second cool-off."""
        now = time.time()
        if self._kafka_offline and (now - self._last_kafka_check < 60):
            return None

        if self._producer is None or self._kafka_offline:
            self._last_kafka_check = now
            try:
                # Import lazily so missing kafka-python never crashes the API
                from kafka import KafkaProducer  # noqa: PLC0415
                self._producer = KafkaProducer(
                    bootstrap_servers=['localhost:9092'],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    acks='all',
                    retries=3,
                    request_timeout_ms=1000,
                    max_block_ms=1000,
                )
                self._kafka_offline = False
                logger.info("Kafka producer connected successfully.")
            except ImportError:
                logger.warning("kafka-python not installed – Kafka events will be skipped.")
                self._kafka_offline = True
                return None
            except Exception as e:
                logger.warning(f"Kafka not available – events will be skipped: {e}. Retrying in 60s.")
                self._producer = None
                self._kafka_offline = True
                return None
        return self._producer

    def emit(self, event: dict):
        payload = {
            'event_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            **event
        }
        producer = self._get_producer()
        if producer is None:
            logger.debug("Skipping event emit – Kafka producer not available.")
            return
        try:
            future = producer.send(self.topic, payload)
            future.get(timeout=5)
        except Exception as e:
            logger.warning(f"Kafka send failed: {e}")
            # Reset producer so next call re-attempts connection
            self._producer = None
            try:
                producer.send(self.dlq, {**payload, 'error': str(e)})
            except Exception:
                pass

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