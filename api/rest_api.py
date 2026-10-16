from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import logging
from dotenv import load_dotenv
import threading

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

from optimization.controller import OptimizationController
from llm_client import LLMClient
from hallucination.detector import HallucinationDetector
from observability.collector import FeedbackCollector
from decision.engine import DecisionEngine

app = FastAPI(title='LLM Optimization Platform', version='2.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = OptimizationController()
llm_client = LLMClient()
detector   = HallucinationDetector(llm_client)
collector  = FeedbackCollector()
engine     = DecisionEngine()

def start_stream_processor():
    try:
        from observability.stream_processor import StreamProcessor
        from observability.db_writer import DBWriter
        db = DBWriter()
        processor = StreamProcessor(db_writer=db)
        logger.info("Starting built-in Stream Processor background thread...")
        processor.run()
    except Exception as e:
        logger.error(f"Stream Processor failed to start: {e}")

threading.Thread(target=start_stream_processor, daemon=True).start()


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response:            str
    strategy:            str
    model:               str
    latency_ms:          float
    tokens:              int
    cost_usd:            float
    hallucination_score: float
    risk_level:          str
    cache_hit:           bool
    explanation:         str = ""
    # Cache hit enrichment fields
    similarity:          Optional[float] = None
    matched_query:       Optional[str]  = None
    cache_source:        Optional[str]  = None
    lookup_latency_ms:   Optional[float] = None
    # Routing metadata & Diagnostics
    query_type:          Optional[str]  = None
    tier:                Optional[str]  = None
    complexity:          Optional[float] = None
    intended_model:      Optional[str]  = None
    intended_provider:   Optional[str]  = None
    actual_model:        Optional[str]  = None
    actual_provider:     Optional[str]  = None
    fallback_occurred:   bool           = False
    fallback_reason:     Optional[str]  = None


@app.on_event("startup")
async def on_startup():
    logger.info("✅ LLM Optimization Platform API v2 started")
    logger.info("📖 Docs available at /docs")


def safe_log_cache_hit(engine, opt):
    try:
        engine.log_query(
            strategy='cache',
            latency_ms=opt['latency_ms'],
            tokens=0,
            cost_usd=0.0,
            hallucination_score=0.0,
            cache_hit=True
        )
    except Exception as e:
        logger.warning(f"Failed to log cache hit to engine: {e}")


def safe_log_query_metrics(engine, collector, query, llm_result, opt, halluc):
    try:
        collector.emit_llm_call(
            query=query,
            response=llm_result['response'],
            strategy=opt['strategy'],
            model=opt['model']['name'],
            latency_ms=llm_result['latency_ms'],
            tokens=llm_result['tokens'],
            cost=llm_result['cost_usd'],
            hallucination_score=halluc['hallucination_score']
        )
    except Exception as e:
        logger.warning(f"Kafka emit failed (non-fatal): {e}")

    try:
        engine.log_query(
            strategy=opt['strategy'],
            latency_ms=llm_result['latency_ms'],
            tokens=llm_result['tokens'],
            cost_usd=llm_result['cost_usd'],
            hallucination_score=halluc['hallucination_score'],
            cache_hit=False
        )
    except Exception as e:
        logger.warning(f"Failed to log query to engine: {e}")


@app.post('/query', response_model=QueryResponse)
async def process_query(req: QueryRequest, background_tasks: BackgroundTasks):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # Precompute query embedding once per request
        query_embedding = controller.cache._embed(req.query)
        opt = controller.process(req.query, engine, query_embedding=query_embedding)

        # ── Cache hit path ────────────────────────────────────────────────────
        if opt.get('cache_hit'):
            background_tasks.add_task(safe_log_cache_hit, engine, opt)

            return QueryResponse(
                response=opt['response'],
                strategy='cache',
                model='cache',
                latency_ms=opt['latency_ms'],
                tokens=0,
                cost_usd=0.0,
                hallucination_score=0.0,
                risk_level='low',
                cache_hit=True,
                explanation="Low Risk: Response served instantly from Semantic Cache.",
                similarity=opt.get('similarity'),
                matched_query=opt.get('matched_query'),
                cache_source=opt.get('cache_source', 'in_memory'),
                lookup_latency_ms=opt.get('lookup_latency_ms'),
                query_type=None,
                tier='cache',
                complexity=None
            )

        # ── LLM call path ─────────────────────────────────────────────────────
        llm_result = llm_client.call(
            opt['system'], opt['prompt'], opt['model'])

        halluc = detector.score(
            req.query, llm_result['response'], opt['model'])

        # Store in cache for future hits in background
        background_tasks.add_task(
            controller.cache.set,
            req.query,
            llm_result['response'],
            query_embedding,
            {'model': opt['model']['name'], 'tier': opt['model'].get('tier'), 'strategy': opt['strategy']}
        )

        background_tasks.add_task(
            safe_log_query_metrics,
            engine,
            collector,
            req.query,
            llm_result,
            opt,
            halluc
        )

        return QueryResponse(
            response=llm_result['response'],
            strategy=opt['strategy'],
            model=llm_result.get('actual_model') or opt['model']['name'],
            latency_ms=llm_result['latency_ms'],
            tokens=llm_result['tokens'],
            cost_usd=llm_result['cost_usd'],
            hallucination_score=halluc['hallucination_score'],
            risk_level=halluc['risk_level'],
            cache_hit=False,
            explanation=halluc['explanation'],
            similarity=None,
            matched_query=None,
            cache_source=None,
            lookup_latency_ms=None,
            query_type=opt['model'].get('query_type'),
            tier=opt['model'].get('tier'),
            complexity=opt['model'].get('complexity'),
            intended_model=llm_result.get('intended_model') or opt['model']['name'],
            intended_provider=llm_result.get('intended_provider') or opt['model'].get('provider'),
            actual_model=llm_result.get('actual_model') or opt['model']['name'],
            actual_provider=llm_result.get('actual_provider') or opt['model'].get('provider'),
            fallback_occurred=llm_result.get('fallback_occurred', False),
            fallback_reason=llm_result.get('fallback_reason')
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get('/metrics')
async def get_metrics():
    try:
        engine.update_weights()
        return engine.get_report()
    except Exception as e:
        logger.warning(f"/metrics error: {e}")
        return {"weights": {}, "recommended": "unknown", "error": str(e)}


@app.get('/cache/stats')
async def cache_stats():
    """Return statistics and entries from the semantic cache."""
    try:
        with controller.cache._lock:
            entries = list(controller.cache._in_memory_cache)

        total = len(entries)
        # Build summary entries (drop embedding to keep payload small)
        summary = []
        for e in entries:
            summary.append({
                'query': e.get('query', ''),
                'normalized_query': e.get('normalized_query', ''),
                'response_preview': e.get('response', '')[:120] + ('...' if len(e.get('response', '')) > 120 else ''),
                'model_metadata': e.get('model_metadata', {}),
                'timestamp': e.get('timestamp', 0),
            })

        return {
            'total_entries': total,
            'threshold': controller.cache.threshold,
            'redis_online': not controller.cache._redis_offline,
            'entries': summary
        }
    except Exception as e:
        logger.error(f"/cache/stats error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete('/cache/clear')
async def cache_clear():
    """Clear all entries from the in-memory semantic cache and persistence file."""
    try:
        with controller.cache._lock:
            cleared = len(controller.cache._in_memory_cache)
            controller.cache._in_memory_cache = []
            controller.cache._save_local_store()

        # Also clear Redis if available
        r = controller.cache._get_redis()
        redis_cleared = 0
        if r is not None:
            try:
                keys = list(r.scan_iter(f'{controller.cache.cache_prefix}*'))
                if keys:
                    redis_cleared = r.delete(*keys)
            except Exception as re:
                logger.warning(f"Redis clear failed: {re}")

        logger.info(f"Cache cleared: {cleared} in-memory entries, {redis_cleared} Redis keys.")
        return {'cleared_in_memory': cleared, 'cleared_redis': redis_cleared, 'status': 'ok'}
    except Exception as e:
        logger.error(f"/cache/clear error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
async def health():
    return {'status': 'ok', 'version': '2.0'}