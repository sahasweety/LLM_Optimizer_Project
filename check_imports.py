from hallucination.detector import get_embedder
print("detector OK")
from optimization.cache_module import CacheModule
print("cache OK")
from optimization.controller import OptimizationController
print("controller OK")
from llm_client import LLMClient
print("llm_client OK")
from api.rest_api import app
print("rest_api OK")
