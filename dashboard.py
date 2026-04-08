import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time
import json
import os

API_URL = "http://127.0.0.1:8081"
HISTORY_FILE = "query_history.json"

st.set_page_config(
    page_title="LLM Optimization Platform",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 LLM Optimization & Observability Platform")
st.markdown("---")


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []


def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


if 'history' not in st.session_state:
    st.session_state.history = load_history()

if 'selected_query' not in st.session_state:
    st.session_state.selected_query = None

# ── SIDEBAR ────────────────────────────────────────────────
def check_api_health() -> str:
    for attempt in range(3):
        try:
            health = requests.get(f"{API_URL}/health", timeout=5)
            if health.status_code == 200:
                return "online"
            return "error"
        except requests.exceptions.ConnectionError:
            if attempt < 2:
                time.sleep(0.5)
        except Exception:
            if attempt < 2:
                time.sleep(0.5)
    return "offline"

HEALTH_CHECK_INTERVAL = 10  # seconds

now = time.time()
last_checked = st.session_state.get("_health_checked_at", 0)
if now - last_checked >= HEALTH_CHECK_INTERVAL:
    st.session_state["_api_status"] = check_api_health()
    st.session_state["_health_checked_at"] = now

api_status = st.session_state.get("_api_status", "offline")

with st.sidebar:
    st.header("⚙️ Platform Status")
    if api_status == "online":
        st.success("✅ API Server Online")
    elif api_status == "error":
        st.warning("⚠️ API Server Error (unexpected status)")
    else:
        st.error("❌ API Server Offline — run: uvicorn api.rest_api:app --host 127.0.0.1 --port 8081")

    if st.button("🔄 Refresh Status", use_container_width=True):
        st.session_state["_health_checked_at"] = 0
        st.rerun()

    st.markdown("---")
    st.header("📊 Session Stats")
    total = len(st.session_state.history)
    cache_hits = sum(1 for h in st.session_state.history if h.get('cache_hit'))
    st.metric("Total Queries", total)
    st.metric("Cache Hits", cache_hits)
    if total > 0:
        st.metric("Cache Hit Rate", f"{(cache_hits/total)*100:.1f}%")

    st.markdown("---")
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.selected_query = None
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        st.success("History cleared!")
        st.rerun()

# ── QUERY SECTION ──────────────────────────────────────────
st.header("💬 Send a Query")

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_area(
        "Enter your query:",
        placeholder="e.g. Analyze supervised vs unsupervised learning...",
        height=100
    )
with col2:
    st.write("")
    st.write("")
    submit = st.button("🚀 Send", use_container_width=True)

if submit and query:
    with st.spinner("Processing query..."):
        try:
            response = requests.post(
                f"{API_URL}/query",
                json={"query": query},
                timeout=60
            )

            if response.status_code == 200:
                data = response.json()
                data['query'] = query
                st.session_state.history.append(data)
                save_history(st.session_state.history)

                st.markdown("---")
                st.header("📝 Response")
                st.write(data['response'])

                st.markdown("---")
                st.header("🧠 Hallucination Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.subheader("Risk Score")
                    score = data['hallucination_score']
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=score,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.3], 'color': "lightgreen"},
                                {'range': [0.3, 0.6], 'color': "yellow"},
                                {'range': [0.6, 1], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.6
                            }
                        }
                    ))
                    fig.update_layout(
                        height=250,
                        margin=dict(t=20, b=20, l=20, r=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.subheader("Risk Level")
                    if score < 0.3:
                        st.success("🟢 LOW RISK")
                        st.write("Response is consistent and confident. Safe to use.")
                    elif score < 0.6:
                        st.warning("🟡 MEDIUM RISK")
                        st.write("Some uncertainty detected. Verify important facts.")
                    else:
                        st.error("🔴 HIGH RISK")
                        st.write("Inconsistent response. Do not rely on this answer.")

                with col3:
                    st.subheader("Query Info")
                    st.info(f"**Strategy:** {data['strategy']}")
                    st.info(f"**Model:** {data['model']}")
                    st.info(f"**Latency:** {data['latency_ms']:.0f}ms")
                    st.info(f"**Cache Hit:** {'✅ Yes' if data['cache_hit'] else '❌ No'}")
                    st.info(f"**Tokens:** {data['tokens']}")
                    st.info(f"**Cost:** ${data['cost_usd']:.6f}")

            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Error: {str(e)}")

elif submit and not query:
    st.warning("Please enter a query first!")

# ── HISTORY ────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.header("📜 Query History")
    st.caption("Click any query row to expand full details and response")

    df_data = []
    for i, h in enumerate(st.session_state.history):
        df_data.append({
            '#': i + 1,
            'Query': h['query'][:55] + '...' if len(h['query']) > 55 else h['query'],
            'Strategy': h['strategy'],
            'Model': h['model'],
            'Latency (ms)': round(h['latency_ms'], 1),
            'Cache Hit': '✅' if h['cache_hit'] else '❌',
            'Hallucination': round(h['hallucination_score'], 3),
            'Risk': h['risk_level'],
            'Tokens': h['tokens'],
            'Cost ($)': round(h['cost_usd'], 6)
        })

    df = pd.DataFrame(df_data)

    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=False,
        on_select="rerun",
        selection_mode="single-row"
    )

    selected_rows = event.selection.rows
    if selected_rows:
        i = selected_rows[0]
        h = st.session_state.history[i]
        st.markdown("---")
        st.markdown(f"**{h['query']}**")
        st.caption(
            f"Strategy: `{h['strategy']}` &nbsp;|&nbsp; "
            f"Model: `{h['model']}` &nbsp;|&nbsp; "
            f"Latency: `{h['latency_ms']:.0f}ms` &nbsp;|&nbsp; "
            f"Cache: `{'✅ Hit' if h['cache_hit'] else '❌ Miss'}` &nbsp;|&nbsp; "
            f"Hallucination: `{h['hallucination_score']:.3f}` &nbsp;|&nbsp; "
            f"Risk: `{h['risk_level'].upper()}` &nbsp;|&nbsp; "
            f"Tokens: `{h['tokens']}` &nbsp;|&nbsp; "
            f"Cost: `${h['cost_usd']:.6f}`"
        )
        st.write(h['response'])
        st.markdown("---")

    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.header("📈 Performance Charts")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("⚡ Latency per Query")
            fig = px.bar(
                df,
                x=df.index,
                y='Latency (ms)',
                color='Strategy',
                title="Response Latency (ms)",
                color_discrete_map={
                    'cache': '#2ECC71',
                    'model_selection': '#3498DB',
                    'prompt+model': '#9B59B6'
                },
                template='plotly_dark'
            )
            fig.update_layout(
                plot_bgcolor='rgba(15,15,30,0.9)',
                paper_bgcolor='rgba(15,15,30,0.9)',
                font_color='white',
                title_font_size=16,
                showlegend=True,
                bargap=0.3
            )
            fig.update_traces(
                marker_line_color='white',
                marker_line_width=1.5,
                opacity=0.85
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🎯 Strategy Distribution")
            strategy_counts = df['Strategy'].value_counts()
            fig = px.pie(
                values=strategy_counts.values,
                names=strategy_counts.index,
                title="Optimization Strategy Used",
                color_discrete_sequence=[
                    '#2ECC71', '#3498DB', '#9B59B6',
                    '#E74C3C', '#F39C12'
                ],
                hole=0.4,
                template='plotly_dark'
            )
            fig.update_layout(
                plot_bgcolor='rgba(15,15,30,0.9)',
                paper_bgcolor='rgba(15,15,30,0.9)',
                font_color='white',
                title_font_size=16
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                marker=dict(line=dict(color='white', width=2)),
                opacity=0.9
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 Hallucination Score Trend")
            fig = px.line(
                df,
                x=df.index,
                y='Hallucination',
                title="Hallucination Score per Query",
                markers=True,
                template='plotly_dark',
                color_discrete_sequence=['#E74C3C']
            )
            fig.add_hrect(
                y0=0, y1=0.3,
                fillcolor="#2ECC71", opacity=0.15,
                annotation_text="Safe Zone",
                annotation_position="top left"
            )
            fig.add_hrect(
                y0=0.3, y1=0.6,
                fillcolor="#F39C12", opacity=0.15,
                annotation_text="Medium Risk",
                annotation_position="top left"
            )
            fig.add_hrect(
                y0=0.6, y1=1.0,
                fillcolor="#E74C3C", opacity=0.15,
                annotation_text="High Risk",
                annotation_position="top left"
            )
            fig.update_layout(
                plot_bgcolor='rgba(15,15,30,0.9)',
                paper_bgcolor='rgba(15,15,30,0.9)',
                font_color='white',
                yaxis_range=[0, 1]
            )
            fig.update_traces(
                line_width=2.5,
                marker_size=10
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("💰 Token Usage per Query")
            fig = px.bar(
                df,
                x=df.index,
                y='Tokens',
                title="Tokens Used per Query",
                color='Tokens',
                color_continuous_scale=[
                    '#1A1A2E', '#16213E',
                    '#0F3460', '#533483',
                    '#E94560'
                ],
                template='plotly_dark'
            )
            fig.update_layout(
                plot_bgcolor='rgba(15,15,30,0.9)',
                paper_bgcolor='rgba(15,15,30,0.9)',
                font_color='white',
                bargap=0.3
            )
            fig.update_traces(
                marker_line_color='white',
                marker_line_width=1,
                opacity=0.9
            )
            st.plotly_chart(fig, use_container_width=True)