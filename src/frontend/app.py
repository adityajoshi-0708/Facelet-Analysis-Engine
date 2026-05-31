"""
app.py — Ahoum Streamlit UI
============================
Run:
    streamlit run app.py
"""

import json
import time
from typing import Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ahoum · Personality Facet Scorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29 0%, #302b63 60%, #24243e 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #e2e8f0 !important; }

/* ── Header ── */
.ahoum-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(102,126,234,0.3);
}
.ahoum-header h1 { margin: 0; font-size: 2.2rem; font-weight: 700; }
.ahoum-header p  { margin: 0.4rem 0 0; opacity: 0.88; font-size: 1rem; }

/* ── Cards ── */
.facet-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    border-left: 5px solid #667eea;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    transition: transform 0.15s;
}
.facet-card:hover { transform: translateX(3px); }
.facet-card h4    { margin: 0 0 0.3rem; color: #2d3748; font-size: 1rem; }
.facet-card p     { margin: 0; color: #718096; font-size: 0.88rem; }

.metric-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
}
.score-pill  { background:#ebf4ff; color:#2b6cb0; }
.conf-pill   { background:#f0fff4; color:#276749; }
.cat-pill    { background:#faf5ff; color:#6b46c1; }

/* ── Score bar ── */
.score-bar-wrap { background:#e2e8f0; border-radius:6px; height:8px; overflow:hidden; margin:4px 0 8px; }
.score-bar-fill { height:100%; border-radius:6px;
    background: linear-gradient(90deg,#667eea,#764ba2); }

/* ── Status dot ── */
.status-ok   { color:#38a169; font-weight:600; }
.status-err  { color:#e53e3e; font-weight:600; }

/* ── Tab override ── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-weight: 500;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,#667eea,#764ba2);
    color: white; border: none; border-radius: 8px;
    padding: 0.5rem 1.5rem; font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; color: white; }

/* ── Evidence quote ── */
.evidence-quote {
    background:#f7fafc; border-left:3px solid #667eea;
    padding:8px 14px; border-radius:0 8px 8px 0;
    font-style:italic; color:#4a5568; font-size:0.9rem;
    margin:6px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = st.sidebar.text_input(
    "API Base URL", value="http://localhost:8000", key="api_base"
).rstrip("/")

CATEGORY_COLORS = {
    "personality":          "#667eea",
    "emotion":              "#ed8936",
    "cognitive":            "#38b2ac",
    "social":               "#48bb78",
    "safety":               "#e53e3e",
    "clinical_health":      "#9f7aea",
    "behavioral_lifestyle": "#f6ad55",
    "spirituality_culture": "#fc8181",
    "other":                "#a0aec0",
}

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: Optional[dict] = None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is the server running?"
    except Exception as e:
        return None, str(e)


def _post(path: str, body: dict, timeout: int = 120):
    try:
        r = requests.post(f"{API_BASE}{path}", json=body, timeout=timeout)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API. Is the server running?"
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("## 🧠 Ahoum")
st.sidebar.markdown("*Personality Facet Intelligence*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "🔍 Retrieve Facets", "💬 Score Turn", "📖 Score Conversation", "📚 Facet Browser"],
)

st.sidebar.markdown("---")
health_data, health_err = _get("/health")
if health_data:
    st.sidebar.markdown(f'<span class="status-ok">● API Online</span>', unsafe_allow_html=True)
    st.sidebar.caption(f"Backend: `{health_data.get('backend','?')}`")
    st.sidebar.caption(f"Facets loaded: **{health_data.get('facets_loaded',0)}**")
    st.sidebar.caption(f"Uptime: {health_data.get('uptime_seconds',0):.0f}s")
else:
    st.sidebar.markdown('<span class="status-err">● API Offline</span>', unsafe_allow_html=True)
    st.sidebar.caption(health_err)

# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

def _score_color(score: int) -> str:
    palette = {1:"#e53e3e",2:"#ed8936",3:"#ecc94b",4:"#48bb78",5:"#38a169"}
    return palette.get(score,"#a0aec0")


def _render_facet_score(fs: dict, show_evidence: bool = True):
    score    = fs.get("score", 0)
    conf     = fs.get("confidence", 0)
    cat      = fs.get("category", "other")
    color    = CATEGORY_COLORS.get(cat, "#a0aec0")
    sc_color = _score_color(score)

    st.markdown(f"""
    <div class="facet-card" style="border-left-color:{color}">
        <h4>{fs.get('facet_name','')}</h4>
        <span class="metric-pill score-pill">Score: {score}/5</span>
        <span class="metric-pill conf-pill">Conf: {conf:.2f}</span>
        <span class="metric-pill cat-pill">{cat}</span>
        <div class="score-bar-wrap">
            <div class="score-bar-fill" style="width:{score/5*100}%;background:{sc_color}"></div>
        </div>
        <p>{fs.get('rationale','')}</p>
    </div>
    """, unsafe_allow_html=True)

    if show_evidence and fs.get("evidence_span"):
        st.markdown(
            f'<div class="evidence-quote">"{fs["evidence_span"]}"</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# Pages
# ===========================================================================

# ── Dashboard ───────────────────────────────────────────────────────────────
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="ahoum-header">
        <h1>🧠 Ahoum</h1>
        <p>Personality & Behavioural Facet Scoring from Conversational Text</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    if health_data:
        col1.metric("Status",         "Online ✓")
        col2.metric("Facets Indexed", health_data.get("facets_loaded", 0))
        col3.metric("LLM Backend",    health_data.get("backend","?"))
        col4.metric("Uptime",         f"{health_data.get('uptime_seconds',0):.0f}s")
    else:
        col1.metric("Status", "Offline ✗")

    st.markdown("---")
    st.markdown("### How it works")
    cols = st.columns(4)
    steps = [
        ("1️⃣ Input",      "Paste a conversation or a single turn"),
        ("2️⃣ Retrieve",   "Hybrid dense+BM25 retrieval finds relevant facets"),
        ("3️⃣ Score",      "LLM scores each facet 1–5 with evidence"),
        ("4️⃣ Insights",   "View scores, confidence, and evidence spans"),
    ]
    for col, (icon, desc) in zip(cols, steps):
        with col:
            st.info(f"**{icon}**\n\n{desc}")

    st.markdown("### Quick Score")
    quick_text = st.text_area(
        "Paste a turn to score instantly",
        placeholder="e.g. I quit my stable job and invested all my savings into a startup...",
        height=100,
    )
    if st.button("Score Now →") and quick_text.strip():
        with st.spinner("Scoring..."):
            data, err = _post("/score/turn", {"text": quick_text, "top_k": 10})
        if err:
            st.error(err)
        elif data:
            st.success(f"Scored in {data.get('latency_ms',0):.0f}ms")
            for fs in data.get("facet_scores", [])[:5]:
                _render_facet_score(fs)


# ── Retrieve Facets ─────────────────────────────────────────────────────────
elif page == "🔍 Retrieve Facets":
    st.markdown("## 🔍 Retrieve Facets")
    st.caption("Find the most relevant personality facets for any text using hybrid retrieval.")

    query = st.text_area("Query text", height=120,
        placeholder="I've been feeling really disconnected lately and don't see the point anymore...")
    top_k = st.slider("Top K results", 5, 50, 20)

    if st.button("Retrieve") and query.strip():
        with st.spinner("Retrieving..."):
            data, err = _post("/retrieve", {"query": query, "top_k": top_k})

        if err:
            st.error(err)
        elif data:
            st.success(f"Retrieved {len(data['results'])} facets in {data['latency_ms']}ms")
            st.markdown("---")

            # Category breakdown
            cats: dict = {}
            for r in data["results"]:
                cats[r.get("category","other")] = cats.get(r.get("category","other"),0) + 1

            cat_cols = st.columns(min(len(cats), 5))
            for col, (cat, count) in zip(cat_cols, sorted(cats.items(), key=lambda x:-x[1])):
                color = CATEGORY_COLORS.get(cat, "#a0aec0")
                col.markdown(
                    f'<div style="background:{color}22;border:1px solid {color};'
                    f'border-radius:8px;padding:8px;text-align:center">'
                    f'<b style="color:{color}">{cat}</b><br><span style="font-size:1.4rem">{count}</span></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            for r in data["results"]:
                color = CATEGORY_COLORS.get(r.get("category","other"), "#a0aec0")
                st.markdown(f"""
                <div class="facet-card" style="border-left-color:{color}">
                    <h4>#{r['rank']} {r['facet_name']}</h4>
                    <span class="metric-pill cat-pill">{r.get('category','')}</span>
                    <span class="metric-pill score-pill">RRF: {r['score']:.6f}</span>
                    <p style="margin-top:4px;font-size:0.8rem;color:#a0aec0">{r['facet_id']}</p>
                </div>
                """, unsafe_allow_html=True)


# ── Score Turn ───────────────────────────────────────────────────────────────
elif page == "💬 Score Turn":
    st.markdown("## 💬 Score Single Turn")
    st.caption("Score one conversation turn against all relevant facets.")

    col1, col2 = st.columns([3, 1])
    with col1:
        turn_text = st.text_area("Turn text", height=150,
            placeholder="I finally told my manager I was burning out. It was terrifying but I had to be honest.")
    with col2:
        speaker = st.selectbox("Speaker", ["user", "assistant"])
        turn_id = st.text_input("Turn ID", "t0")
        top_k   = st.slider("Top K", 5, 40, 20)

    if st.button("Score Turn") and turn_text.strip():
        with st.spinner("Scoring — this may take 15–30s on first run..."):
            data, err = _post(
                "/score/turn",
                {"text": turn_text, "turn_id": turn_id, "speaker": speaker, "top_k": top_k},
                timeout=180,
            )

        if err:
            st.error(err)
        elif data:
            scores = data.get("facet_scores", [])
            st.success(f"✓ {len(scores)} facets scored in {data.get('latency_ms',0):.0f}ms")
            st.markdown("---")

            # Top facet highlight
            if scores:
                top = scores[0]
                st.markdown(f"""
                <div style="background:linear-gradient(135deg,#667eea,#764ba2);
                    border-radius:12px;padding:1.5rem;color:white;margin-bottom:1rem">
                    <h3 style="margin:0">Top Facet: {top['facet_name']}</h3>
                    <p style="margin:0.5rem 0 0;opacity:0.9">
                        Score <b>{top['score']}/5</b> · Confidence <b>{top['confidence']:.2f}</b>
                    </p>
                    <p style="margin:0.5rem 0 0;font-style:italic;opacity:0.85">
                        "{top.get('evidence_span','')}"
                    </p>
                </div>
                """, unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["📊 All Scores", "📋 Raw JSON"])
            with tab1:
                for fs in scores:
                    _render_facet_score(fs)
            with tab2:
                st.json(data)


# ── Score Conversation ───────────────────────────────────────────────────────
elif page == "📖 Score Conversation":
    st.markdown("## 📖 Score Full Conversation")
    st.caption("Enter a multi-turn conversation and score every turn.")

    conv_id = st.text_input("Conversation ID", "conv_001")

    st.markdown("### Turns")
    st.caption("Add turns below. Click '+ Add Turn' to add more.")

    if "turns" not in st.session_state:
        st.session_state.turns = [
            {"speaker": "user", "text": "I quit my stable job yesterday to start a company."},
            {"speaker": "assistant", "text": "That's a huge leap! What made you decide to do it now?"},
            {"speaker": "user", "text": "I've been miserable for two years. The risk felt worth it."},
        ]

    for i, turn in enumerate(st.session_state.turns):
        c1, c2, c3 = st.columns([1, 5, 0.5])
        with c1:
            st.session_state.turns[i]["speaker"] = st.selectbox(
                "Speaker", ["user","assistant"], key=f"spk_{i}",
                index=0 if turn["speaker"]=="user" else 1,
            )
        with c2:
            st.session_state.turns[i]["text"] = st.text_input(
                "Text", value=turn["text"], key=f"txt_{i}", label_visibility="collapsed"
            )
        with c3:
            if st.button("✕", key=f"del_{i}") and len(st.session_state.turns) > 1:
                st.session_state.turns.pop(i)
                st.rerun()

    if st.button("+ Add Turn"):
        st.session_state.turns.append({"speaker":"user","text":""})
        st.rerun()

    st.markdown("---")
    if st.button("🚀 Score Conversation") and any(t["text"].strip() for t in st.session_state.turns):
        payload = {
            "conversation_id": conv_id,
            "turns": [t for t in st.session_state.turns if t["text"].strip()],
        }
        with st.spinner("Scoring conversation — may take 30–60s..."):
            data, err = _post("/score", payload, timeout=300)

        if err:
            st.error(err)
        elif data:
            st.success(
                f"✓ Scored {data['n_turns']} turns in {data['total_latency_ms']:.0f}ms"
            )

            # Summary bar chart
            st.markdown("### 📊 Facet Summary (mean score across turns)")
            summary = data.get("summary", {})
            if summary:
                import pandas as pd
                df_sum = pd.DataFrame(
                    list(summary.items())[:15], columns=["Facet","Mean Score"]
                ).sort_values("Mean Score", ascending=True)
                st.bar_chart(df_sum.set_index("Facet"))

            st.markdown("---")
            st.markdown("### Turn-by-Turn Results")
            for turn_data in data.get("turns", []):
                tid = turn_data["turn_id"]
                scores = turn_data.get("facet_scores", [])
                with st.expander(f"Turn {tid} — {len(scores)} facets scored"):
                    for fs in scores[:8]:
                        _render_facet_score(fs)

            with st.expander("📋 Raw JSON"):
                st.json(data)


# ── Facet Browser ────────────────────────────────────────────────────────────
elif page == "📚 Facet Browser":
    st.markdown("## 📚 Facet Browser")
    st.caption("Explore every facet in the enriched catalogue.")

    col1, col2 = st.columns([2,1])
    with col2:
        category_filter = st.selectbox(
            "Filter by category",
            ["All","personality","emotion","cognitive","social","safety",
             "clinical_health","behavioral_lifestyle","spirituality_culture","other"],
        )
    with col1:
        search_term = st.text_input("Search facet name", placeholder="e.g. empathy, risk...")

    page_num  = st.number_input("Page", min_value=1, value=1, step=1)
    page_size = 30

    params: dict = {"page": page_num, "page_size": page_size}
    if category_filter != "All":
        params["category"] = category_filter

    data, err = _get("/facets", params=params)

    if err:
        st.error(err)
    elif data:
        total = data.get("total",0)
        pages = data.get("pages",1)
        st.caption(f"**{total}** facets · Page {page_num}/{pages}")

        facets = data.get("facets",[])
        if search_term:
            facets = [f for f in facets if search_term.lower() in f["facet_name"].lower()]

        if not facets:
            st.info("No facets match your search.")
        else:
            cols = st.columns(2)
            for i, f in enumerate(facets):
                color = CATEGORY_COLORS.get(f.get("category","other"),"#a0aec0")
                with cols[i % 2]:
                    if st.button(f"🔎 {f['facet_name']}", key=f"btn_{f['facet_id']}",
                                 use_container_width=True):
                        st.session_state["selected_facet"] = f["facet_id"]

                    st.markdown(f"""
                    <div class="facet-card" style="border-left-color:{color};margin-top:-0.5rem">
                        <span class="metric-pill cat-pill">{f.get('category','')}</span>
                        <p style="margin-top:6px">{f.get('description','')[:120]}…</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Detail panel
        if "selected_facet" in st.session_state:
            fid = st.session_state["selected_facet"]
            detail, derr = _get(f"/facets/{fid}")
            if detail:
                st.markdown("---")
                st.markdown(f"### 🔬 {detail.get('facet_name','')}")
                color = CATEGORY_COLORS.get(detail.get("category","other"),"#a0aec0")
                st.markdown(
                    f'<span class="metric-pill cat-pill">{detail.get("category","")}</span> '
                    f'<code>{detail.get("facet_id","")}</code>',
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Description:** {detail.get('description','')}")

                c1, c2 = st.columns(2)
                with c1:
                    pos = detail.get("positive_indicators",[])
                    if isinstance(pos, str):
                        try: pos = json.loads(pos)
                        except: pos = []
                    st.markdown("**✅ Positive indicators**")
                    for p in pos[:5]: st.markdown(f"- {p}")
                with c2:
                    neg = detail.get("negative_indicators",[])
                    if isinstance(neg, str):
                        try: neg = json.loads(neg)
                        except: neg = []
                    st.markdown("**❌ Negative indicators**")
                    for n in neg[:5]: st.markdown(f"- {n}")

                st.markdown("**Score anchors**")
                anchor_cols = st.columns(5)
                for lvl, col in enumerate(anchor_cols, 1):
                    text = detail.get(f"score_{lvl}_anchor","")
                    col.markdown(
                        f'<div style="background:#f7fafc;border-radius:8px;'
                        f'padding:8px;text-align:center;font-size:0.8rem">'
                        f'<b style="color:{_score_color(lvl)}">{lvl}</b><br>{text[:80]}</div>',
                        unsafe_allow_html=True,
                    )