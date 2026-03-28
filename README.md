# 🤖 LLM Optimization & Observability Platform

A production-grade backend platform that intelligently optimizes Large Language Model queries using caching, prompt engineering, and dynamic model selection — while monitoring every response for hallucinations in real time.

---

## 📌 Overview

Most LLM applications are slow, expensive, and unreliable. This platform solves all three problems:

- **Slow** → Redis semantic cache returns repeated queries in ~50ms instead of 2000ms
- **Expensive** → Smart model routing uses cheap models for simple queries, powerful models only when needed
- **Unreliable** → Every response is scored for hallucination risk before being returned

---

## ✨ Features

- **3 Optimization Strategies** — Cache, Model Selection, Prompt+Model
- **4 LLM Providers** — Groq (llama), Google Gemini, Anthropic Claude, OpenAI GPT-4o-mini
- **Hallucination Detection** — Self-consistency checks + confidence scoring → 0.0–1.0 risk score
- **Real-time Observability** — Kafka event streaming → Stream Processor → PostgreSQL
- **Decision Engine** — Pareto optimization + epsilon-greedy RL auto-selects best strategy
- **Streamlit Dashboard** — Live charts, query history, hallucination gauge

---

## 🏗️ Architecture
```
User Query
    ↓
Optimization Controller
    ├── Cache Module (Redis · similarity ≥ 0.80 → instant return)
    ├── Model Selector (complexity score → fast/balanced/powerful/expert)
    └── Prompt Module (task detection → QA/code/analysis/summary)
    ↓
Groq / Google Gemini / Claude / GPT-4o-mini
    ↓
Hallucination Detector (confidence + consistency → risk score)
    ↓
Kafka → Stream Processor → PostgreSQL → Decision Engine
    ↓
Streamlit Dashboard
```

---

## 🤖 Model Selection Logic

| Complexity Score | Model | Provider |
|-----------------|-------|----------|
| 0.00 – 0.18 | llama-3.1-8b-instant | Groq |
| 0.18 – 0.32 | gemini-2.5-flash-lite | Google |
| 0.32 – 0.48 | claude-3-haiku | OpenRouter |
| 0.48 – 1.00 | gpt-4o-mini | OpenRouter |

---

## 🧠 Hallucination Detection

| Score | Risk Level | Meaning |
|-------|-----------|---------|
| 0.00 – 0.25 | 🟢 LOW | Safe to use |
| 0.25 – 0.50 | 🟡 MEDIUM | Verify important facts |
| 0.50 – 1.00 | 🔴 HIGH | Do not rely on this response |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | Python, FastAPI |
| Dashboard | Streamlit, Plotly |
| LLM Providers | Groq, Google Gemini, OpenRouter |
| Cache | Redis |
| Event Streaming | Apache Kafka |
| Database | PostgreSQL |
| ML Embeddings | sentence-transformers |
| Infrastructure | Docker, Docker Compose |

---

## 🚀 Setup & Run

### Prerequisites
- Python 3.11+
- Docker Desktop
- API keys for Groq, Google, OpenRouter

### Installation
```bash
# Clone the repo
git clone https://github.com/sahasweety/LLM_Optimizer_Project.git
cd LLM_Optimizer_Project

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the root folder:
```
GROQ_API_KEY=your-groq-key
GOOGLE_API_KEY=your-google-key
OPENROUTER_API_KEY=your-openrouter-key
DATABASE_URL=postgresql://admin:secret@localhost:5433/llmplatform
```

### Run
```bash
# Start Docker services
docker-compose up -d

# Start API server
python -m uvicorn api.rest_api:app --host 127.0.0.1 --port 8081 --reload

# Start Streamlit dashboard
streamlit run dashboard.py
```

Open browser at `http://localhost:8501`

---

## 📊 Dashboard

- Send queries and see real-time results
- Hallucination risk gauge (0.0 – 1.0)
- Query history with click-to-expand details
- 4 performance charts: Latency, Strategy Distribution, Hallucination Trend, Token Usage

---

## 📁 Project Structure
```
llm-platform/
├── optimization/
│   ├── cache_module.py       # Redis semantic cache
│   ├── prompt_module.py      # Prompt optimization
│   ├── model_selector.py     # Dynamic model selection
│   └── controller.py        # Optimization controller
├── hallucination/
│   └── detector.py          # Hallucination detection
├── observability/
│   ├── collector.py         # Kafka event emitter
│   ├── stream_processor.py  # Stream aggregator
│   └── db_writer.py         # PostgreSQL writer
├── decision/
│   └── engine.py            # Decision engine (RL)
├── api/
│   └── rest_api.py          # FastAPI endpoints
├── dashboard.py             # Streamlit dashboard
├── docker-compose.yml       # Docker services
└── requirements.txt
```

---

## 👤 Author

**Sweety Saha**
- GitHub: [@sahasweety](https://github.com/sahasweety)