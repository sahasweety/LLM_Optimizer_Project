from observability.stream_processor import StreamProcessor
from observability.db_writer import DBWriter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("Starting Stream Processor...")
print("Listening to Kafka topic: llm-events")
print("Writing summaries every 60 seconds...")
print("Press Ctrl+C to stop")

db = DBWriter()
processor = StreamProcessor(db_writer=db)
processor.run()