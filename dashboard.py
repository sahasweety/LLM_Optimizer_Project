import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json
import os
from datetime import datetime

API_URL = os.getenv("API_URL", "http://127.0.0.1:8081")
HISTORY_FILE = "query_history.json"

st.set_page_config(
    page_title="LLM Optimization Platform",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(15, 52, 96, 0.5);
    }
    .metric-label { color: #8892b0; font-size: 0.75rem; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #ccd6f6; font-size: 1.8rem; font-weight: 700; margin: 0.2rem 0; }
    .metric-sub   { color: #64ffda; font-size: 0.75rem; }

    .routing-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-fast     { background: #0f3460; color: #64ffda; border: 1px solid #64ffda; }
    .badge-balanced { background: #3d1a78; color: #c084fc; border: 1px solid #c084fc; }
    .badge-powerful { background: #7c2d12; color: #fb923c; border: 1px solid #fb923c; }
    .badge-expert   { background: #14532d; color: #4ade80; border: 1px solid #4ade80; }
    .badge-cache    { background: #1e3a5f; color: #38bdf8; border: 1px solid #38bdf8; }

    .cache-hit-banner {
        background: linear-gradient(90deg, #064e3b, #065f46);
        border-left: 4px solid #34d399;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        color: #d1fae5;
        font-size: 0.9rem;
    }

    .stTab > div > div > div > div > button {
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Utility helpers ─────────────────────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
    except Exception:
        pass


def tier_badge(tier: str) -> str:
    classes = {
        'fast': 'badge-fast', 'balanced': 'badge-balanced',
        'powerful': 'badge-powerful', 'expert': 'badge-expert', 'cache': 'badge-cache'
    }
    cls = classes.get(str(tier).lower(), 'badge-fast')
    return f'<span class="routing-badge {cls}">{str(tier).upper()}</span>'


def risk_color(risk: str) -> str:
    return {'low': '🟢', 'medium': '🟡', 'high': '🔴'}.get(risk, '⚪')


# ── Session state init ──────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = load_history()
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = None


# ── API health check (cached 10 s) ─────────────────────────────────────────────
def check_api_health() -> tuple[str, str]:
    target_url = API_URL.rstrip('/')
    for attempt in range(3):
        try:
            h = requests.get(f"{target_url}/health", timeout=5)
            if h.status_code == 200:
                return "online", f"Connected to {target_url}"
            return "error", f"HTTP {h.status_code} from {target_url}"
        except requests.exceptions.ConnectionError:
            if attempt < 2:
                time.sleep(0.5)
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
    return "offline", f"Cannot reach {target_url}"


HEALTH_INTERVAL = 10
now = time.time()
if now - st.session_state.get("_health_checked_at", 0) >= HEALTH_INTERVAL:
    status, detail = check_api_health()
    st.session_state["_api_status"]     = status
    st.session_state["_api_detail"]     = detail
    st.session_state["_health_checked_at"] = now

api_status = st.session_state.get("_api_status", "offline")
api_detail = st.session_state.get("_api_detail", f"Target: {API_URL}")


# ── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 LLM Platform")
    st.markdown("---")
    st.header("⚙️ API Status")
    if api_status == "online":
        st.success(f"✅ Online\n\n`{API_URL}`")
    elif api_status == "error":
        st.warning(f"⚠️ Server Error\n\n{api_detail}")
    else:
        st.error(f"❌ Offline\n\n`{API_URL}`")
        if "127.0.0.1" in API_URL or "localhost" in API_URL:
            st.caption("Start backend:\n`uvicorn api.rest_api:app --host 127.0.0.1 --port 8081`")
        else:
            st.caption("Backend waking up or `API_URL` env var needs updating.")

    if st.button("🔄 Refresh Status", use_container_width=True):
        st.session_state["_health_checked_at"] = 0
        st.rerun()

    st.markdown("---")
    st.header("📊 Session Stats")
    total       = len(st.session_state.history)
    cache_hits  = sum(1 for h in st.session_state.history if h.get('cache_hit'))
    llm_calls   = total - cache_hits
    total_cost  = sum(h.get('cost_usd', 0) for h in st.session_state.history)
    total_tokens = sum(h.get('tokens', 0) for h in st.session_state.history)

    st.metric("Total Queries",  total)
    st.metric("Cache Hits",     cache_hits)
    st.metric("LLM Calls",      llm_calls)
    if total > 0:
        st.metric("Cache Hit Rate", f"{(cache_hits/total)*100:.1f}%")
    st.metric("Total Cost",     f"${total_cost:.4f}")
    st.metric("Total Tokens",   f"{total_tokens:,}")

    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.selected_query = None
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.success("History cleared!")
        st.rerun()


# ── MAIN HEADER ─────────────────────────────────────────────────────────────────
st.title("🤖 LLM Routing, Caching & Evaluation Platform")
st.caption("Real-world semantic caching · Multi-provider LLM routing · Hallucination evaluation")
st.markdown("---")


# ── TABS ─────────────────────────────────────────────────────────────────────────
tab_query, tab_routing, tab_cache, tab_analytics = st.tabs([
    "💬 Query",
    "🔀 Routing Decisions",
    "🗄️ Cache Manager",
    "📈 Analytics"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — QUERY
# ════════════════════════════════════════════════════════════════════════════════
with tab_query:
    st.header("💬 Send a Query")

    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_area(
            "Enter your query:",
            placeholder="e.g. Explain the difference between supervised and unsupervised learning...",
            height=110,
            key="main_query_input"
        )
    with col2:
        st.write("")
        st.write("")
        submit = st.button("🚀 Send", use_container_width=True, key="main_submit")

    if submit and query:
        with st.spinner("🔄 Processing..."):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": query},
                    timeout=90
                )

                if response.status_code == 200:
                    data = response.json()
                    data['query'] = query
                    data['timestamp'] = time.time()
                    st.session_state.history.append(data)
                    save_history(st.session_state.history)

                    # ── Cache hit banner ───────────────────────────────────────
                    if data.get('cache_hit'):
                        sim   = data.get('similarity', 0)
                        mq    = data.get('matched_query', 'N/A')
                        src   = data.get('cache_source', 'in_memory')
                        lat   = data.get('lookup_latency_ms', 0)
                        st.markdown(f"""
                        <div class="cache-hit-banner">
                            ⚡ <strong>Semantic Cache HIT</strong> —
                            Similarity: <strong>{sim:.3f}</strong> &nbsp;|&nbsp;
                            Source: <strong>{src}</strong> &nbsp;|&nbsp;
                            Lookup: <strong>{lat:.1f}ms</strong><br/>
                            <span style="font-size:0.82rem;opacity:0.8;">Matched: <em>"{mq}"</em></span>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Response ───────────────────────────────────────────────
                    st.markdown("---")
                    st.subheader("📝 Response")
                    st.write(data['response'])

                    # ── Metrics row ────────────────────────────────────────────
                    st.markdown("---")
                    st.subheader("📊 Response Metrics")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Strategy",  data['strategy'].replace('_', ' ').title())
                    m2.metric("Model",     data['model'])
                    m3.metric("Latency",   f"{data['latency_ms']:.0f} ms")
                    m4.metric("Tokens",    data['tokens'])
                    m5.metric("Cost",      f"${data['cost_usd']:.6f}")

                    if data.get('tier') and not data.get('cache_hit'):
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Tier",       data.get('tier', 'N/A').upper())
                        r2.metric("Query Type", data.get('query_type', 'N/A').replace('_', ' ').title())
                        r3.metric("Complexity", f"{data.get('complexity', 0):.2f}")

                    # ── Hallucination gauge ────────────────────────────────────
                    st.markdown("---")
                    st.subheader("🧠 Hallucination Analysis")
                    g1, g2, g3 = st.columns(3)

                    with g1:
                        score = data['hallucination_score']
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            title={'text': "Risk Score", 'font': {'color': '#ccd6f6'}},
                            gauge={
                                'axis': {'range': [0, 1], 'tickcolor': '#8892b0'},
                                'bar': {'color': "#e94560"},
                                'bgcolor': '#1a1a2e',
                                'steps': [
                                    {'range': [0, 0.3],  'color': '#064e3b'},
                                    {'range': [0.3, 0.6],'color': '#78350f'},
                                    {'range': [0.6, 1],  'color': '#7f1d1d'}
                                ],
                                'threshold': {
                                    'line': {'color': "#e94560", 'width': 3},
                                    'thickness': 0.75,
                                    'value': 0.6
                                }
                            },
                            number={'font': {'color': '#ccd6f6'}}
                        ))
                        fig.update_layout(
                            height=240,
                            margin=dict(t=30, b=10, l=10, r=10),
                            paper_bgcolor='rgba(0,0,0,0)',
                            font_color='#ccd6f6'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with g2:
                        st.subheader("Risk Level")
                        risk = data['risk_level']
                        if risk == 'low':
                            st.success("🟢 LOW RISK\n\nConsistent and confident. Safe to use.")
                        elif risk == 'medium':
                            st.warning("🟡 MEDIUM RISK\n\nSome uncertainty. Verify important facts.")
                        else:
                            st.error("🔴 HIGH RISK\n\nInconsistent response. Do not rely on this.")

                    with g3:
                        st.subheader("Explanation")
                        st.info(data.get('explanation', ''))

                else:
                    try:
                        err = response.json().get("detail", response.text)
                    except Exception:
                        err = response.text
                    st.error(f"API Error ({response.status_code}): {err}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out after 90 s. The backend may be cold-starting or overloaded.")
            except requests.exceptions.ConnectionError:
                st.error(f"🔌 Could not connect to `{API_URL}`. Start the backend first.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    elif submit and not query:
        st.warning("Please enter a query first!")

    # ── Compact history table in query tab ─────────────────────────────────────
    if st.session_state.history:
        st.markdown("---")
        st.subheader("📜 Recent Queries")

        df_data = []
        for i, h in enumerate(reversed(st.session_state.history[-20:])):
            df_data.append({
                '#': len(st.session_state.history) - i,
                'Query':         h['query'][:60] + ('…' if len(h['query']) > 60 else ''),
                'Strategy':      h['strategy'],
                'Model':         h['model'],
                'Latency (ms)':  round(h['latency_ms'], 1),
                'Cache':         '✅' if h.get('cache_hit') else '❌',
                'Hallucination': round(h.get('hallucination_score', 0), 3),
                'Risk':          h.get('risk_level', 'low').upper(),
                'Tokens':        h.get('tokens', 0),
                'Cost ($)':      round(h.get('cost_usd', 0), 6),
            })

        df = pd.DataFrame(df_data)
        event = st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        selected_rows = event.selection.rows
        if selected_rows:
            rev_idx = selected_rows[0]
            h = list(reversed(st.session_state.history[-20:]))[rev_idx]
            st.markdown("---")
            st.markdown(f"**{h['query']}**")
            st.caption(
                f"Strategy: `{h['strategy']}` &nbsp;|&nbsp; "
                f"Model: `{h['model']}` &nbsp;|&nbsp; "
                f"Latency: `{h['latency_ms']:.0f}ms` &nbsp;|&nbsp; "
                f"Cache: `{'✅ Hit' if h.get('cache_hit') else '❌ Miss'}` &nbsp;|&nbsp; "
                f"Hallucination: `{h.get('hallucination_score',0):.3f}` &nbsp;|&nbsp; "
                f"Risk: `{h.get('risk_level','low').upper()}`"
            )
            st.write(h['response'])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — ROUTING DECISIONS
# ════════════════════════════════════════════════════════════════════════════════
with tab_routing:
    st.header("🔀 LLM Routing Decisions")
    st.caption("Every query shows which model tier was selected, why, and how the decision engine scored each strategy.")

    if not st.session_state.history:
        st.info("No routing history yet. Send some queries in the Query tab.")
    else:
        # Strategy weights from decision engine
        try:
            metrics_resp = requests.get(f"{API_URL}/metrics", timeout=5)
            if metrics_resp.status_code == 200:
                metrics_data = metrics_resp.json()
                weights      = metrics_data.get('weights', {})
                recommended  = metrics_data.get('recommended', 'N/A')

                st.subheader("🎯 Decision Engine Strategy Weights")
                wc1, wc2, wc3, wc4 = st.columns(4)
                for col, (strat, w) in zip([wc1, wc2, wc3, wc4], weights.items()):
                    col.metric(strat.replace('_',' ').title(), f"{w:.3f}")
                st.caption(f"Recommended strategy: **{recommended}**")
                st.markdown("---")
        except Exception:
            st.caption("(Decision engine metrics unavailable)")

        # Per-query routing table
        st.subheader("📋 Per-Query Routing Log")
        routing_rows = []
        for i, h in enumerate(st.session_state.history):
            routing_rows.append({
                '#':            i + 1,
                'Query':        h['query'][:55] + ('…' if len(h['query']) > 55 else ''),
                'Cache Hit':    '✅ Yes' if h.get('cache_hit') else '❌ No',
                'Similarity':   f"{h.get('similarity', 0) or 0:.3f}" if h.get('cache_hit') else '—',
                'Tier':         h.get('tier', 'cache' if h.get('cache_hit') else 'N/A'),
                'Query Type':   h.get('query_type', '—') or '—',
                'Complexity':   f"{h.get('complexity', 0) or 0:.2f}" if not h.get('cache_hit') else '—',
                'Model':        h['model'],
                'Strategy':     h['strategy'],
                'Latency (ms)': round(h['latency_ms'], 1),
                'Cost ($)':     round(h.get('cost_usd', 0), 6),
            })

        rdf = pd.DataFrame(routing_rows)
        st.dataframe(rdf, use_container_width=True, hide_index=True)

        st.markdown("---")
        rc1, rc2 = st.columns(2)

        with rc1:
            st.subheader("🏷️ Model Tier Distribution")
            tier_data = {}
            for h in st.session_state.history:
                t = h.get('tier', 'cache' if h.get('cache_hit') else 'fast') or 'fast'
                tier_data[t] = tier_data.get(t, 0) + 1

            if tier_data:
                fig = px.pie(
                    values=list(tier_data.values()),
                    names=list(tier_data.keys()),
                    title="Model Tier Distribution",
                    color_discrete_map={
                        'fast': '#0f3460', 'balanced': '#3d1a78',
                        'powerful': '#7c2d12', 'expert': '#14532d', 'cache': '#1e3a5f'
                    },
                    hole=0.45,
                    template='plotly_dark'
                )
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
                st.plotly_chart(fig, use_container_width=True)

        with rc2:
            st.subheader("📦 Query Type Distribution")
            qtype_data = {}
            for h in st.session_state.history:
                qt = h.get('query_type', 'cache' if h.get('cache_hit') else 'factual') or 'factual'
                qtype_data[qt] = qtype_data.get(qt, 0) + 1

            if qtype_data:
                fig = px.bar(
                    x=list(qtype_data.keys()),
                    y=list(qtype_data.values()),
                    title="Queries by Detected Type",
                    color=list(qtype_data.keys()),
                    template='plotly_dark',
                    labels={'x': 'Query Type', 'y': 'Count'}
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(15,15,30,0.9)',
                    font_color='white',
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — CACHE MANAGER
# ════════════════════════════════════════════════════════════════════════════════
with tab_cache:
    st.header("🗄️ Semantic Cache Manager")
    st.caption("Inspect, browse, and manage the in-memory semantic cache.")

    c_col1, c_col2 = st.columns([3, 1])
    with c_col1:
        if st.button("🔃 Refresh Cache Stats", use_container_width=True, key="refresh_cache"):
            st.session_state["_cache_stats_ts"] = 0  # force refresh

    with c_col2:
        if st.button("🗑️ Clear Entire Cache", use_container_width=True, key="clear_cache_btn", type="primary"):
            try:
                r = requests.delete(f"{API_URL}/cache/clear", timeout=10)
                if r.status_code == 200:
                    d = r.json()
                    st.success(f"Cache cleared! ({d.get('cleared_in_memory',0)} in-memory, {d.get('cleared_redis',0)} Redis entries)")
                    st.session_state["_cache_stats_ts"] = 0
                else:
                    st.error(f"Failed to clear cache: {r.text}")
            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Fetch cache stats
    try:
        cs_resp = requests.get(f"{API_URL}/cache/stats", timeout=10)
        if cs_resp.status_code == 200:
            cs = cs_resp.json()
            total_cached = cs.get('total_entries', 0)
            threshold    = cs.get('threshold', 0.82)
            redis_online = cs.get('redis_online', False)
            entries      = cs.get('entries', [])

            # Stats row
            ca1, ca2, ca3, ca4 = st.columns(4)
            ca1.metric("Cached Entries",    total_cached)
            ca2.metric("Similarity Threshold", f"{threshold:.2f}")
            ca3.metric("Redis",            "🟢 Online" if redis_online else "🔴 Offline")
            ca4.metric("Persistence File", "✅ Active" if os.path.exists("cache_store.json") else "—")

            st.markdown("---")

            if entries:
                st.subheader(f"📂 Cache Entries ({total_cached})")
                cache_table = []
                for i, e in enumerate(entries):
                    mm = e.get('model_metadata', {})
                    ts = e.get('timestamp', 0)
                    dt = datetime.fromtimestamp(ts).strftime('%H:%M:%S') if ts else '—'
                    cache_table.append({
                        '#':              i + 1,
                        'Query':          e.get('query', '')[:60] + ('…' if len(e.get('query','')) > 60 else ''),
                        'Normalized':     e.get('normalized_query', '')[:50] + ('…' if len(e.get('normalized_query','')) > 50 else ''),
                        'Model':          mm.get('model', '—'),
                        'Tier':           mm.get('tier', '—'),
                        'Cached At':      dt,
                        'Response Preview': e.get('response_preview', ''),
                    })

                cdf = pd.DataFrame(cache_table)
                event_c = st.dataframe(
                    cdf,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row"
                )
                sel_c = event_c.selection.rows
                if sel_c:
                    selected_entry = entries[sel_c[0]]
                    st.markdown("---")
                    st.markdown(f"**Original query:** {selected_entry.get('query', '')}")
                    st.markdown(f"**Normalized:**     `{selected_entry.get('normalized_query', '')}`")
                    mm = selected_entry.get('model_metadata', {})
                    if mm:
                        st.markdown(f"**Model metadata:** {json.dumps(mm)}")
                    st.markdown("**Response preview:**")
                    st.info(selected_entry.get('response_preview', ''))
            else:
                st.info("The cache is empty. Send a few queries to populate it!")

        else:
            st.error(f"Could not fetch cache stats: {cs_resp.status_code}")
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to `{API_URL}`. Start the backend first.")
    except Exception as e:
        st.error(f"Cache stats error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.header("📈 Performance Analytics")

    if not st.session_state.history:
        st.info("No data yet. Send some queries in the Query tab.")
    else:
        history = st.session_state.history

        # ── Summary KPIs ───────────────────────────────────────────────────────
        total        = len(history)
        cache_hits   = sum(1 for h in history if h.get('cache_hit'))
        llm_calls    = total - cache_hits
        avg_latency  = sum(h.get('latency_ms', 0) for h in history) / total
        cache_lat    = [h.get('latency_ms', 0) for h in history if h.get('cache_hit')]
        llm_lat      = [h.get('latency_ms', 0) for h in history if not h.get('cache_hit')]
        total_cost   = sum(h.get('cost_usd', 0) for h in history)
        total_tokens = sum(h.get('tokens', 0) for h in history)
        avg_halluc   = sum(h.get('hallucination_score', 0) for h in history) / total

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Total Queries",   total)
        k2.metric("Cache Hit Rate",  f"{(cache_hits/total)*100:.1f}%", delta=f"{cache_hits} hits")
        k3.metric("Avg Latency",     f"{avg_latency:.0f} ms",
                  delta=f"Cache: {sum(cache_lat)/len(cache_lat):.0f}ms" if cache_lat else None)
        k4.metric("Total Tokens",    f"{total_tokens:,}")
        k5.metric("Total Cost",      f"${total_cost:.4f}")
        k6.metric("Avg Hallucination", f"{avg_halluc:.3f}")

        st.markdown("---")
        row1_c1, row1_c2 = st.columns(2)

        with row1_c1:
            st.subheader("⚡ Latency per Query")
            df_lat = pd.DataFrame([{
                'Index': i,
                'Latency (ms)': h.get('latency_ms', 0),
                'Strategy': h.get('strategy', 'unknown'),
                'Cache': 'Cache Hit' if h.get('cache_hit') else 'LLM Call'
            } for i, h in enumerate(history)])
            fig = px.bar(
                df_lat, x='Index', y='Latency (ms)', color='Cache',
                title="Response Latency per Query",
                color_discrete_map={'Cache Hit': '#34d399', 'LLM Call': '#60a5fa'},
                template='plotly_dark'
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,15,30,0.9)',
                font_color='white', bargap=0.25
            )
            st.plotly_chart(fig, use_container_width=True)

        with row1_c2:
            st.subheader("🎯 Strategy Distribution")
            strat_counts = {}
            for h in history:
                s = h.get('strategy', 'unknown')
                strat_counts[s] = strat_counts.get(s, 0) + 1

            fig = px.pie(
                values=list(strat_counts.values()),
                names=list(strat_counts.keys()),
                title="Optimization Strategy Used",
                color_discrete_sequence=['#34d399', '#60a5fa', '#c084fc', '#fb923c', '#f87171'],
                hole=0.4,
                template='plotly_dark'
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            fig.update_traces(textposition='inside', textinfo='percent+label',
                              marker=dict(line=dict(color='white', width=2)))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        row2_c1, row2_c2 = st.columns(2)

        with row2_c1:
            st.subheader("🧠 Hallucination Score Trend")
            fig = px.line(
                x=list(range(len(history))),
                y=[h.get('hallucination_score', 0) for h in history],
                title="Hallucination Score per Query",
                markers=True, template='plotly_dark',
                color_discrete_sequence=['#f87171'],
                labels={'x': 'Query Index', 'y': 'Score'}
            )
            fig.add_hrect(y0=0,   y1=0.3, fillcolor="#064e3b", opacity=0.2,
                          annotation_text="Safe", annotation_position="top left")
            fig.add_hrect(y0=0.3, y1=0.6, fillcolor="#78350f", opacity=0.2,
                          annotation_text="Medium", annotation_position="top left")
            fig.add_hrect(y0=0.6, y1=1.0, fillcolor="#7f1d1d", opacity=0.2,
                          annotation_text="High Risk", annotation_position="top left")
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,15,30,0.9)',
                font_color='white', yaxis_range=[0, 1]
            )
            st.plotly_chart(fig, use_container_width=True)

        with row2_c2:
            st.subheader("💰 Cost & Token Usage")
            costs  = [h.get('cost_usd', 0)  for h in history]
            tokens = [h.get('tokens', 0)     for h in history]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='Tokens', x=list(range(len(history))), y=tokens,
                yaxis='y', marker_color='#60a5fa', opacity=0.8
            ))
            fig.add_trace(go.Scatter(
                name='Cost ($)', x=list(range(len(history))), y=costs,
                yaxis='y2', mode='lines+markers',
                line=dict(color='#f59e0b', width=2), marker=dict(size=6)
            ))
            fig.update_layout(
                title="Token Usage vs Cost",
                xaxis_title="Query Index",
                yaxis=dict(title='Tokens', color='#60a5fa'),
                yaxis2=dict(title='Cost ($)', color='#f59e0b', overlaying='y', side='right'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,15,30,0.9)',
                font_color='white',
                legend=dict(orientation='h', y=1.1),
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── Latency comparison box chart ────────────────────────────────────────
        if cache_lat and llm_lat:
            st.markdown("---")
            st.subheader("📦 Latency Distribution: Cache vs LLM")
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=cache_lat, name='Cache Hits',
                marker_color='#34d399', boxmean=True
            ))
            fig.add_trace(go.Box(
                y=llm_lat, name='LLM Calls',
                marker_color='#60a5fa', boxmean=True
            ))
            fig.update_layout(
                title="Response Time Distribution",
                yaxis_title="Latency (ms)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(15,15,30,0.9)',
                font_color='white',
                template='plotly_dark'
            )
            st.plotly_chart(fig, use_container_width=True)