from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import os
import logging
from dotenv import load_dotenv

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

app = FastAPI(title='LLM Optimization Platform', version='1.0')

controller = OptimizationController()
llm_client = LLMClient()
detector   = HallucinationDetector(llm_client)
collector  = FeedbackCollector()
engine     = DecisionEngine()


class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response:           str
    strategy:           str
    model:              str
    latency_ms:         float
    tokens:             int
    cost_usd:           float
    hallucination_score: float
    risk_level:         str
    cache_hit:          bool
    explanation:        str = ""


@app.on_event("startup")
async def on_startup():
    logger.info("✅ LLM Optimization Platform API started on http://127.0.0.1:8081")
    logger.info("📖 Docs available at http://127.0.0.1:8081/docs")


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
    # Emit event to Kafka (non-blocking; fails gracefully)
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

    # Log query metrics to DecisionEngine
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
                explanation="Low Risk: Response served instantly from Semantic Cache."
            )

        # ── LLM call path ─────────────────────────────────────────────────────
        llm_result = llm_client.call(
            opt['system'], opt['prompt'], opt['model'])

        halluc = detector.score(
            req.query, llm_result['response'], opt['model'])

        # Store in cache for future hits in background
        background_tasks.add_task(controller.cache.set, req.query, llm_result['response'], query_embedding)

        # Log query metrics and emit to Kafka in background
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
            model=opt['model']['name'],
            latency_ms=llm_result['latency_ms'],
            tokens=llm_result['tokens'],
            cost_usd=llm_result['cost_usd'],
            hallucination_score=halluc['hallucination_score'],
            risk_level=halluc['risk_level'],
            cache_hit=False,
            explanation=halluc['explanation']
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


@app.get('/health')
async def health():
    return {'status': 'ok'}