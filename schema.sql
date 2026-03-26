CREATE TABLE IF NOT EXISTS llm_events (
    id                  SERIAL PRIMARY KEY,
    event_id            UUID NOT NULL,
    timestamp           TIMESTAMPTZ DEFAULT NOW(),
    query               TEXT,
    strategy            VARCHAR(50),
    model               VARCHAR(100),
    latency_ms          FLOAT,
    tokens              INT,
    cost_usd            FLOAT,
    hallucination_score FLOAT,
    risk_level          VARCHAR(10)
);

CREATE TABLE IF NOT EXISTS window_summaries (
    id                SERIAL PRIMARY KEY,
    window_start      TIMESTAMPTZ,
    strategy          VARCHAR(50),
    event_count       INT,
    avg_latency_ms    FLOAT,
    avg_tokens        FLOAT,
    total_cost_usd    FLOAT,
    avg_hallucination FLOAT
);

CREATE INDEX IF NOT EXISTS idx_strategy  ON llm_events(strategy);
CREATE INDEX IF NOT EXISTS idx_timestamp ON llm_events(timestamp);