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
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        text-align: center;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .metric-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(15, 52, 96, 0.4);
    }
    .metric-label { color: #8892b0; font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 2px; }
    .metric-value { color: #ccd6f6; font-size: 1.15rem; font-weight: 700; margin: 0; line-height: 1.2; }
    .metric-sub   { color: #64ffda; font-size: 0.68rem; margin-top: 2px; }

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

    .cache-status-bar {
        border-radius: 8px;
        padding: 0.55rem 0.9rem;
        margin: 0.4rem 0 0.8rem 0;
        font-size: 0.85rem;
        display: flex;
        flex-direction: column;
        gap: 3px;
    }
    .cache-status-hit {
        background: linear-gradient(90deg, #064e3b, #065f46);
        border-left: 4px solid #34d399;
        color: #d1fae5;
    }
    .cache-status-miss {
        background: linear-gradient(90deg, #1e293b, #0f172a);
        border-left: 4px solid #38bdf8;
        color: #e2e8f0;
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


def render_query_result(data: dict, show_query_header: bool = False):
    """
    Renders Query -> Answer -> Compact Performance Metrics -> Inline Hallucination Analysis.
    Every query response has its own dedicated, accurate analysis rendered directly below it.
    """
    if show_query_header and data.get('query'):
        st.markdown(f"#### ❓ Query: *{data['query']}*")

    # 1. Compact Cache Status Bar (Hit or Miss)
    if data.get('cache_hit'):
        sim = data.get('similarity')
        mq  = data.get('matched_query')
        src = data.get('cache_source', 'in_memory')
        lat = data.get('lookup_latency_ms', 0)
        
        sim_str = f" &nbsp;|&nbsp; Similarity: <strong>{sim:.3f}</strong>" if sim is not None else ""
        mq_str  = f'<br/><span style="font-size:0.8rem;opacity:0.85;">Matched Query: <em>"{mq}"</em></span>' if mq else ""
        
        st.markdown(f"""
        <div class="cache-status-bar cache-status-hit">
            <div>⚡ <strong>CACHE HIT</strong> — Answer served from {src} ({lat:.1f}ms){sim_str}</div>
            {mq_str}
        </div>
        """, unsafe_allow_html=True)
    else:
        selected_model = data.get('actual_model') or data.get('model', 'N/A')
        intended_model = data.get('intended_model') or selected_model
        actual_provider = (data.get('actual_provider') or '').upper()
        intended_provider = (data.get('intended_provider') or actual_provider).upper()
        actual_latency = data.get('latency_ms', 0)
        tier = data.get('tier')
        tier_info = f" ({tier.upper()} tier)" if tier else ""
        
        fallback_flag = data.get('fallback_occurred', False)
        fallback_note = ""
        if fallback_flag:
            fallback_reason = data.get('fallback_reason', 'Provider error')
            fallback_note = f"""<br/><span style="color:#fbbf24;font-size:0.8rem;">⚠️ Fallback triggered: Intended {intended_model} ({intended_provider}) failed → handled by {selected_model} ({actual_provider}). Reason: {fallback_reason}</span>"""
            
        provider_disp = f"[{actual_provider}] " if actual_provider else ""
        st.markdown(f"""
        <div class="cache-status-bar cache-status-miss">
            <div>🌐 <strong>CACHE MISS</strong> — Routed to <strong>{provider_disp}{selected_model}</strong>{tier_info} &nbsp;|&nbsp; Latency: <strong>{actual_latency:.0f} ms</strong></div>
            {fallback_note}
        </div>
        """, unsafe_allow_html=True)

    # 2. Response (Main focus)
    st.markdown("### 📝 Response")
    st.write(data.get('response', ''))

    # 3. Compact Performance Metrics Row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        lat_val = f"{data.get('latency_ms', 0):.0f} ms"
        lat_sub = f"Lookup: {data.get('lookup_latency_ms', 0):.1f}ms" if data.get('cache_hit') else "API latency"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latency</div>
            <div class="metric-value">{lat_val}</div>
            <div class="metric-sub">{lat_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        c_val = "CACHE HIT" if data.get('cache_hit') else "CACHE MISS"
        c_sub = f"Sim: {data.get('similarity', 0):.2f}" if data.get('cache_hit') and data.get('similarity') is not None else "Fresh API call"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Cache Status</div>
            <div class="metric-value" style="color: {'#34d399' if data.get('cache_hit') else '#38bdf8'};">{c_val}</div>
            <div class="metric-sub">{c_sub}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m3:
        raw_model = data.get('model', 'unknown')
        short_model = raw_model.split('/')[-1] if '/' in raw_model else raw_model
        strat = data.get('strategy', 'auto').replace('_', ' ').title()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model</div>
            <div class="metric-value" style="font-size: 0.95rem; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;" title="{raw_model}">{short_model}</div>
            <div class="metric-sub">{strat}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_m4:
        cost = data.get('cost_usd', 0)
        tokens = data.get('tokens', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Est. Cost</div>
            <div class="metric-value">${cost:.5f}</div>
            <div class="metric-sub">{tokens:,} tokens</div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Per-Query Hallucination Analysis (Restored directly below the answer)
    st.markdown("#### 🧠 Hallucination Analysis")
    h_col1, h_col2, h_col3 = st.columns([1, 1.1, 1.9])
    
    raw_risk = data.get('risk_level')
    risk = (raw_risk if isinstance(raw_risk, str) else 'low').lower()
    score = float(data.get('hallucination_score', 0.0) or 0.0)
    
    with h_col1:
        # Compact gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={'text': "Risk Score", 'font': {'color': '#ccd6f6', 'size': 12}},
            gauge={
                'axis': {'range': [0, 1], 'tickcolor': '#8892b0', 'tickwidth': 1, 'dtick': 0.5},
                'bar': {'color': "#e94560"},
                'bgcolor': '#1a1a2e',
                'steps': [
                    {'range': [0, 0.25], 'color': '#064e3b'},
                    {'range': [0.25, 0.5], 'color': '#78350f'},
                    {'range': [0.5, 1.0], 'color': '#7f1d1d'}
                ],
                'threshold': {
                    'line': {'color': "#e94560", 'width': 2},
                    'thickness': 0.75,
                    'value': 0.5
                }
            },
            number={'font': {'color': '#ccd6f6', 'size': 20}, 'valueformat': '.3f'}
        ))
        fig.update_layout(
            height=135,
            margin=dict(t=25, b=5, l=10, r=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ccd6f6'
        )
        st.plotly_chart(fig, use_container_width=True)

    with h_col2:
        st.markdown("<div style='font-size:0.75rem;font-weight:600;color:#8892b0;text-transform:uppercase;margin-bottom:4px;'>Risk Assessment</div>", unsafe_allow_html=True)
        if data.get('cache_hit'):
            st.success("🟢 LOW RISK\n\nVerified Cached Result")
            st.caption("Assessment reused from cached origin.")
        elif risk == 'low':
            st.success("🟢 LOW RISK\n\nConsistent & Confident")
            st.caption("No fabrication or speculation detected.")
        elif risk == 'medium':
            st.warning("🟡 MEDIUM RISK\n\nPotential Inconsistency")
            st.caption("Minor divergence or hedging language.")
        else:
            st.error("🔴 HIGH RISK\n\nHigh Inconsistency")
            st.caption("High variance across verification runs.")

    with h_col3:
        st.markdown("<div style='font-size:0.75rem;font-weight:600;color:#8892b0;text-transform:uppercase;margin-bottom:4px;'>Evaluation Findings</div>", unsafe_allow_html=True)
        exp = data.get('explanation') or "Evaluation complete."
        st.info(exp)
        if not data.get('cache_hit') and (data.get('tier') or data.get('query_type')):
            intended_p = (data.get('intended_provider') or '').upper() if isinstance(data.get('intended_provider'), str) else ''
            actual_p = (data.get('actual_provider') or '').upper() if isinstance(data.get('actual_provider'), str) else ''
            prov_text = f" | Provider: `{actual_p}`" if actual_p else ""
            fb_text = " (Fallback)" if data.get('fallback_occurred') else ""
            complexity_val = data.get('complexity')
            comp_str = f"{complexity_val:.2f}" if isinstance(complexity_val, (int, float)) else "N/A"
            st.caption(f"**Routing:** Tier: `{data.get('tier', 'N/A')}` | Type: `{data.get('query_type', 'N/A')}` | Complexity: `{comp_str}`{prov_text}{fb_text}")


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
tab_query, tab_routing, tab_analytics = st.tabs([
    "💬 Query",
    "🔀 Routing Decisions",
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

                    # ── Render Query -> Answer -> Metrics -> Per-Query Hallucination Analysis ──
                    render_query_result(data, show_query_header=False)

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
                'Risk':          (h.get('risk_level') or 'low').upper() if isinstance(h.get('risk_level') or 'low', str) else 'LOW',
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
            selected_item = list(reversed(st.session_state.history[-20:]))[rev_idx]
            st.markdown("---")
            # Render selected historical query with its own exact answer, metrics, and hallucination analysis
            render_query_result(selected_item, show_query_header=True)


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
                'Model':        h.get('actual_model') or h['model'],
                'Provider':     (h.get('actual_provider') or '—').upper() if not h.get('cache_hit') else 'CACHE',
                'Fallback':     '⚠️ Yes' if h.get('fallback_occurred') else 'No',
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
# TAB 3 — ANALYTICS
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