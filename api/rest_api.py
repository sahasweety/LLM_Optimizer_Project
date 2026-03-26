from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

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
    response: str
    strategy: str
    model: str
    latency_ms: float
    tokens: int
    cost_usd: float
    hallucination_score: float
    risk_level: str
    cache_hit: bool

@app.post('/query', response_model=QueryResponse)
async def process_query(req: QueryRequest):
    opt = controller.process(req.query)

    if opt['cache_hit']:
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

    llm_result = llm_client.call(
        opt['system'], opt['prompt'], opt['model'])

    halluc = detector.score(
        req.query, llm_result['response'], opt['model'])

    controller.cache.set(req.query, llm_result['response'])

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

@app.get('/metrics')
async def get_metrics():
    engine.update_weights()
    return engine.get_report()

@app.get('/health')
async def health():
    return {'status': 'ok'}