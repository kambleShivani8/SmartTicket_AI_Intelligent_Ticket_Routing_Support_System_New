import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import re
import pickle
from scipy.sparse import hstack
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ticket AI Assistant",
    layout="wide",
    page_icon="🎫",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #0e1117 !important;
    color: #cdd9e5;
}
[data-testid="stHeader"] { background: #0d1117; border-bottom: 0.5px solid #21262d; }
[data-testid="stToolbar"] { display: none; }

.topbar {
    background: #0d1117;
    border-bottom: 0.5px solid #21262d;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin: -1rem -1rem 1.5rem -1rem;
}
.topbar-title { font-size: 16px; font-weight: 600; color: #e6edf3; }
.topbar-badge {
    font-size: 11px; background: #1a2a1a; color: #3fb950;
    border: 0.5px solid #238636; border-radius: 12px; padding: 2px 10px;
}
.col-header {
    font-size: 12px; color: #8b949e; display: flex; align-items: center;
    gap: 6px; border-bottom: 0.5px solid #21262d;
    padding-bottom: 10px; margin-bottom: 14px; font-weight: 500;
    letter-spacing: 0.04em;
}
.bubble-user {
    background: #1c2d3d; color: #cdd9e5; font-size: 13px;
    line-height: 1.55; padding: 10px 14px;
    border-radius: 12px 12px 2px 12px;
    max-width: 82%; margin-left: auto; margin-bottom: 10px;
    border: 0.5px solid #1f6feb33;
}
.bubble-ai-wrap { display: flex; gap: 8px; align-items: flex-start; margin-bottom: 10px; }
.ai-avatar {
    width: 28px; height: 28px; border-radius: 50%;
    background: #1c2d3d; border: 0.5px solid #1f6feb;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; font-size: 14px;
}
.bubble-ai {
    background: #161b22; border: 0.5px solid #21262d; color: #cdd9e5;
    font-size: 13px; line-height: 1.6; padding: 10px 14px;
    border-radius: 12px 12px 12px 2px; max-width: 88%;
}
.ai-field { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; font-size: 12px; }
.ai-label { color: #8b949e; min-width: 74px; }
.ai-val { color: #cdd9e5; font-weight: 500; }
.pill {
    font-size: 11px; font-weight: 600; padding: 2px 9px;
    border-radius: 10px; display: inline-block;
}
.pill-critical { background: #3d1a1a; color: #f87171; border: 0.5px solid #7f1d1d; }
.pill-high     { background: #3d2a0a; color: #fbbf24; border: 0.5px solid #78350f; }
.pill-medium   { background: #0d2d45; color: #60a5fa; border: 0.5px solid #1e3a5f; }
.pill-low      { background: #0d2d1a; color: #4ade80; border: 0.5px solid #14532d; }
.decision-ok   { font-size: 11px; color: #3fb950; margin-top: 6px; }
.decision-warn { font-size: 11px; color: #d29922; margin-top: 6px; }
.icard {
    background: #161b22; border: 0.5px solid #21262d;
    border-radius: 8px; padding: 10px 13px; margin-bottom: 9px;
}
.icard-label {
    font-size: 10px; color: #6e7681; text-transform: uppercase;
    letter-spacing: 0.07em; margin-bottom: 4px;
}
.icard-val { font-size: 13px; color: #cdd9e5; font-weight: 500; }
.conf-track { background: #21262d; border-radius: 3px; height: 5px; margin-top: 7px; }
.conf-fill  { height: 5px; border-radius: 3px; background: #1f6feb; }
.etable { width: 100%; font-size: 12px; border-collapse: collapse; margin-top: 4px; }
.etable td { padding: 3px 0; }
.etable td:first-child { color: #8b949e; }
.etable td:last-child  { text-align: right; color: #79c0ff; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>

<div class="topbar">
    <span style="font-size:20px;">🎫</span>
    <span class="topbar-title">Ticket AI Assistant</span>
    <span class="topbar-badge">● AI online</span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CATEGORY_SEVERITY = {
    "Server & Infrastructure Issue": 1.0,
    "Security Issue": 0.95,
    "Network & VPN Issue": 0.8,
    "Account & Access Issue": 0.7,
    "Software & App Issue": 0.6,
    "Hardware Issue": 0.55,
    "Billing & Payment Issue": 0.5,
    "Setup & Configuration Issue": 0.4,
    "Shipping & Delivery Issue": 0.35,
    "Cancellation Request": 0.25,
    "General Support": 0.2,
}

RULE_KEYWORDS = {
    "Server & Infrastructure Issue": ["server", "database", "outage", "infrastructure", "down"],
    "Security Issue": ["breach", "hack", "ransomware", "malware", "security"],
    "Network & VPN Issue": ["vpn", "network", "internet", "firewall"],
    "Account & Access Issue": ["login", "password", "account", "access"],
    "Software & App Issue": ["app", "software", "crash", "bug", "error"],
    "Hardware Issue": ["hardware", "printer", "laptop", "monitor"],
    "Billing & Payment Issue": ["billing", "invoice", "payment", "refund"],
}

SOLUTION_MAP = {
    "Server & Infrastructure Issue": ["Restarted server cluster", "Fixed database connection"],
    "Network & VPN Issue": ["Reset VPN config", "Checked firewall rules"],
    "Account & Access Issue": ["Reset credentials", "Unlocked account"],
    "Security Issue": ["Rotated credentials", "Escalated to Security Team"],
    "Software & App Issue": ["Reinstalled software", "Applied latest patch"],
}

TEAM_MAP = {
    "Network & VPN Issue": "Network Team",
    "Server & Infrastructure Issue": "Infrastructure Team",
    "Billing & Payment Issue": "Finance Team",
    "Hardware Issue": "IT Support Team",
    "Software & App Issue": "Application Support Team",
    "Security Issue": "Security Team",
    "Account & Access Issue": "Identity Team",
}


# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "priority_model.pkl"), "rb") as f:
        priority_model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "priority_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    return vectorizer, model, priority_model, scaler


vectorizer, model, priority_model, scaler = load_models()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def assign_label(text):
    t = str(text).lower()

    for category, keywords in RULE_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return category

    return "General Support"

def build_priority_features(texts, categories, vectorizer, scaler):
    tfidf_feats = vectorizer.transform(texts)

    extra = []

    for text, cat in zip(texts, categories):
        t = str(text).lower()
        words = t.split()

        length = min(len(words) / 60.0, 1.0)
        severity = CATEGORY_SEVERITY.get(cat, 0.2)

        urgency = sum([
            1 if w in t else 0
            for w in ["urgent", "critical", "down", "not working"]
        ]) / 4

        extra.append([length, severity, urgency])

    extra = np.array(extra)
    extra_scaled = scaler.transform(extra)

    return hstack([tfidf_feats, extra_scaled])

def get_team(label, priority=None):
    base = TEAM_MAP.get(label, "Support Team")

    if priority == "Critical":
        return f"Senior {base}"

    return base

def extract_entities(text):
    t = text.lower()

    module = None
    platform = None
    issue = None

    if "server" in t:
        module = "Server"

    if "database" in t:
        module = "Database"

    if "vpn" in t:
        module = "VPN"

    if "aws" in t:
        platform = "AWS"

    if "windows" in t:
        platform = "Windows"

    if "crash" in t:
        issue = "Crash"

    elif "slow" in t:
        issue = "Performance"

    elif "error" in t:
        issue = "Error"

    return {
        "Module": module,
        "Platform": platform,
        "Issue": issue
    }

# ─────────────────────────────────────────────────────────────
# FAST PREDICTION
# ─────────────────────────────────────────────────────────────
def predict_full(ticket: str):

    vec = vectorizer.transform([ticket])

    model_label = model.predict(vec)[0]

    probs = model.predict_proba(vec)[0]

    confidence = round(float(max(probs)) * 100, 1)

    rule_label = assign_label(ticket)

    category = (
        rule_label
        if rule_label != "General Support"
        else model_label
    )

    X_p = build_priority_features(
        [ticket],
        [category],
        vectorizer,
        scaler
    )

    priority = priority_model.predict(X_p)[0]

    team = get_team(category, priority)

    routing_conf = round(confidence * 0.95, 1)

    solutions = SOLUTION_MAP.get(
        category,
        ["No known fix"]
    )

    entities = extract_entities(ticket)

    decision = (
        "Auto Assigned"
        if confidence > 75
        else "Human Review"
    )

    similar = [
        "Previous ticket with similar issue",
        "Server outage reported earlier",
        "Related support request found"
    ]

    return {
        "category": category,
        "priority": priority,
        "team": team,
        "confidence": confidence,
        "routing_conf": routing_conf,
        "similar": similar,
        "solutions": solutions,
        "entities": entities,
        "decision": decision,
    }

# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
PILL_CSS = {
    "Critical": "pill pill-critical",
    "High": "pill pill-high",
    "Medium": "pill pill-medium",
    "Low": "pill pill-low",
}

def priority_pill(p: str):
    cls = PILL_CSS.get(p, "pill")
    return f'<span class="{cls}">{p.upper()}</span>'

def render_ai_bubble(r: dict):

    dec_cls = (
        "decision-ok"
        if r["decision"] == "Auto Assigned"
        else "decision-warn"
    )

    dec_icon = (
        "✔"
        if r["decision"] == "Auto Assigned"
        else "⚠"
    )

    return f"""
<div class="bubble-ai-wrap">
  <div class="ai-avatar">🤖</div>
  <div class="bubble-ai">
    <div class="ai-field"><span class="ai-label">Category</span>
      <span class="ai-val">{r['category']}</span></div>

    <div class="ai-field"><span class="ai-label">Priority</span>
      {priority_pill(r['priority'])}</div>

    <div class="ai-field"><span class="ai-label">Team</span>
      <span class="ai-val">{r['team']}</span></div>

    <div class="ai-field"><span class="ai-label">Confidence</span>
      <span class="ai-val">{r['confidence']}%</span></div>

    <div class="{dec_cls}">
      {dec_icon} {r['decision']}
    </div>
  </div>
</div>
"""

def render_insights(r: dict):

    conf_w = int(r["confidence"])

    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Priority</div>
      {priority_pill(r['priority'])}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Category</div>
      <div class="icard-val">{r['category']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Assigned Team</div>
      <div class="icard-val">{r['team']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Confidence</div>
      <div class="icard-val">{r['confidence']}%</div>
      <div class="conf-track">
        <div class="conf-fill" style="width:{conf_w}%"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    ent = r["entities"]

    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Entities</div>
      <table class="etable">
        <tr><td>Module</td><td>{ent.get('Module') or 'N/A'}</td></tr>
        <tr><td>Platform</td><td>{ent.get('Platform') or 'N/A'}</td></tr>
        <tr><td>Issue</td><td>{ent.get('Issue') or 'N/A'}</td></tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
chat_col, info_col = st.columns([2, 1], gap="medium")

with chat_col:

    st.markdown(
        '<div class="col-header">💬 Conversation</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.messages:
        st.markdown("""
        <div class="bubble-ai-wrap">
          <div class="ai-avatar">🤖</div>
          <div class="bubble-ai">
            Hello! Describe your support issue.
          </div>
        </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.messages:

        if msg["role"] == "user":
            st.markdown(
                f'<div class="bubble-user">{msg["content"]}</div>',
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                msg["content"],
                unsafe_allow_html=True
            )

    user_input = st.chat_input("Describe your issue...")

    if user_input:

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        st.markdown(
            f'<div class="bubble-user">{user_input}</div>',
            unsafe_allow_html=True
        )

        with st.spinner("Analyzing ticket..."):
            result = predict_full(user_input)

        st.session_state.last_result = result

        ai_html = render_ai_bubble(result)

        st.markdown(ai_html, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_html
        })

with info_col:

    st.markdown(
        '<div class="col-header">📊 Ticket Insights</div>',
        unsafe_allow_html=True
    )

    if st.session_state.last_result:
        render_insights(st.session_state.last_result)

    else:
        st.markdown("""
        <div style="text-align:center;color:#484f58;font-size:13px;padding:40px 0;">
          <div style="font-size:28px;margin-bottom:8px;">📥</div>
          Enter a ticket to see insights
        </div>
        """, unsafe_allow_html=True)
