from fastapi import FastAPI, HTTPException
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


@app.on_event("startup")
async def on_startup():
    logger.info("✅ LLM Optimization Platform API started on http://127.0.0.1:8081")
    logger.info("📖 Docs available at http://127.0.0.1:8081/docs")


@app.post('/query', response_model=QueryResponse)
async def process_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        opt = controller.process(req.query)

        # ── Cache hit path ────────────────────────────────────────────────────
        if opt.get('cache_hit'):
            return QueryResponse(
                response=opt['response'],
                strategy='cache',
                model='cache',
                latency_ms=opt['latency_ms'],
                tokens=0,
                cost_usd=0.0,
                hallucination_score=0.0,
                risk_level='low',
                cache_hit=True
            )

        # ── LLM call path ─────────────────────────────────────────────────────
        llm_result = llm_client.call(
            opt['system'], opt['prompt'], opt['model'])

        halluc = detector.score(
            req.query, llm_result['response'], opt['model'])

        # Store in cache for future hits
        controller.cache.set(req.query, llm_result['response'])

        # Emit event to Kafka (non-blocking; fails gracefully)
        try:
            collector.emit_llm_call(
                query=req.query,
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

        return QueryResponse(
            response=llm_result['response'],
            strategy=opt['strategy'],
            model=opt['model']['name'],
            latency_ms=llm_result['latency_ms'],
            tokens=llm_result['tokens'],
            cost_usd=llm_result['cost_usd'],
            hallucination_score=halluc['hallucination_score'],
            risk_level=halluc['risk_level'],
            cache_hit=False
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