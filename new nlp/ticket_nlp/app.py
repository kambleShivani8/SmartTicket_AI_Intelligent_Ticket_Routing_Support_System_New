import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import random
import numpy as np
import pickle
from datetime import datetime
from scipy.sparse import hstack
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ticket AI Assistant",
    layout="wide",
    page_icon="🎫",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #080c12 !important;
    color: #c9d1d9;
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"]     { background: #080c12; border-bottom: 1px solid #161b22; }
[data-testid="stToolbar"]    { display: none; }
[data-testid="stDecoration"] { display: none; }
#MainMenu, footer            { visibility: hidden; }

/* ── topbar ── */
.topbar {
    background: #0d1117;
    border-bottom: 1px solid #161b22;
    padding: 10px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
    margin: -1rem -1rem 1.5rem -1rem;
}
.topbar-title  { font-size: 15px; font-weight: 600; color: #e6edf3; letter-spacing: .02em; }
.topbar-badge  {
    font-size: 11px; background: #0d2010; color: #3fb950;
    border: 1px solid #1a4726; border-radius: 20px; padding: 2px 10px;
    display: flex; align-items: center; gap: 5px;
}
.badge-dot {
    width: 6px; height: 6px; border-radius: 50%; background: #3fb950;
    animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }
.topbar-sep { margin-left: auto; font-size: 11px; color: #484f58; }

/* ── column headers ── */
.col-header {
    font-size: 11px; font-weight: 600; color: #6e7681;
    letter-spacing: .08em; text-transform: uppercase;
    border-bottom: 1px solid #161b22;
    padding-bottom: 10px; margin-bottom: 16px;
}

/* ── insight cards ── */
.icard {
    background: #0d1117; border: 1px solid #1c2128;
    border-radius: 10px; padding: 11px 14px; margin-bottom: 10px;
    transition: border-color .2s;
}
.icard:hover { border-color: #30363d; }
.icard-label {
    font-size: 10px; color: #6e7681; text-transform: uppercase;
    letter-spacing: .08em; margin-bottom: 6px; font-weight: 600;
}
.icard-val { font-size: 13px; color: #e6edf3; font-weight: 500; }

/* ── priority pills ── */
.pill          { font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 20px; display: inline-block; letter-spacing: .04em; }
.pill-critical { background:#2d0f0f; color:#f87171; border:1px solid #6b1a1a; }
.pill-high     { background:#2d1f0a; color:#fbbf24; border:1px solid #7c4a00; }
.pill-medium   { background:#0c1e30; color:#60a5fa; border:1px solid #1e3a5f; }
.pill-low      { background:#0c2018; color:#4ade80; border:1px solid #14532d; }

/* ── confidence bars ── */
.conf-wrap  { margin-top: 6px; }
.conf-row   { display:flex; justify-content:space-between; font-size:11px; margin-bottom:3px; }
.conf-label { color:#6e7681; }
.conf-pct   { color:#79c0ff; font-family:'JetBrains Mono',monospace; }
.conf-track { background:#161b22; border-radius:4px; height:4px; }
.conf-fill  { height:4px; border-radius:4px; }

/* ── entity table ── */
.etable               { width:100%; border-collapse:collapse; font-size:12px; margin-top:2px; }
.etable tr            { border-bottom:1px solid #161b22; }
.etable tr:last-child { border-bottom:none; }
.etable td            { padding:4px 2px; }
.etable td:first-child { color:#6e7681; width:60%; }
.etable td:last-child  { color:#79c0ff; font-family:'JetBrains Mono',monospace; text-align:right; }

/* ── decisions ── */
.dec-ok   { display:inline-flex; align-items:center; gap:5px; font-size:11px; color:#3fb950; background:#0d2010; border:1px solid #1a4726; border-radius:6px; padding:3px 9px; }
.dec-warn { display:inline-flex; align-items:center; gap:5px; font-size:11px; color:#d29922; background:#201800; border:1px solid #4b3200; border-radius:6px; padding:3px 9px; }

/* ── solution list ── */
.sol-list    { list-style:none; padding:0; margin:0; }
.sol-list li { font-size:12px; color:#c9d1d9; padding:4px 0; border-bottom:1px solid #161b22; display:flex; align-items:flex-start; gap:7px; }
.sol-list li:last-child { border-bottom:none; }
.sol-dot { color:#1f6feb; flex-shrink:0; margin-top:2px; }

/* ── empty state ── */
.empty-state { text-align:center; color:#30363d; font-size:13px; padding:50px 20px; border:1px dashed #161b22; border-radius:10px; margin-top:10px; }
.empty-icon  { font-size:32px; margin-bottom:10px; }

/* ── chat bubbles ── */
.bubble-user {
    background:#1c2d3d; border:1px solid #1f6feb33; color:#cdd9e5;
    font-size:13px; line-height:1.6; padding:10px 14px;
    border-radius:14px 14px 2px 14px;
    max-width:78%; margin-left:auto; margin-bottom:14px; word-break:break-word;
}
.bubble-ai-row { display:flex; gap:10px; align-items:flex-start; margin-bottom:14px; }
.ai-avatar {
    width:30px; height:30px; border-radius:50%;
    background:#0d1f2d; border:1px solid #1f6feb55;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0; font-size:15px;
}
.bubble-ai {
    background:#0d1117; border:1px solid #1c2128; color:#cdd9e5;
    font-size:13px; line-height:1.7; padding:14px 16px;
    border-radius:14px 14px 14px 2px; max-width:88%;
    font-family:'JetBrains Mono',monospace;
    white-space:pre-wrap; word-break:break-word;
}
.bubble-welcome {
    background:#0d1117; border:1px solid #1c2128; color:#8b949e;
    font-size:13px; line-height:1.6; padding:12px 16px;
    border-radius:14px 14px 14px 2px;
}

/* ── thank-you bubble ── */
.bubble-thanks {
    background:#091209; border:1px solid #238636; color:#cdd9e5;
    font-size:13px; line-height:1.7; padding:14px 16px;
    border-radius:14px 14px 14px 2px; max-width:88%; word-break:break-word;
}
.ty-head { color:#3fb950; font-weight:600; font-size:13px; margin-bottom:5px; }
.ty-body { color:#8b949e; font-size:12px; line-height:1.65; }
.ty-hi   { color:#cdd9e5; }
.ticket-chip {
    display:inline-block; margin-top:8px;
    font-size:10px; font-family:'JetBrains Mono',monospace;
    color:#484f58; background:#0d1117;
    border:1px solid #21262d; border-radius:5px; padding:2px 8px;
}

/* ── chat input ── */
[data-testid="stChatInput"] textarea {
    background:#0d1117 !important;
    border:1px solid #21262d !important;
    color:#cdd9e5 !important;
    border-radius:10px !important;
    font-size:13px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color:#1f6feb !important;
    box-shadow:0 0 0 3px #1f6feb18 !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar        { width:5px; height:5px; }
::-webkit-scrollbar-track  { background:#080c12; }
::-webkit-scrollbar-thumb  { background:#21262d; border-radius:4px; }
::-webkit-scrollbar-thumb:hover { background:#30363d; }
</style>

<div class="topbar">
    <span style="font-size:20px">🎫</span>
    <span class="topbar-title">Ticket AI Assistant</span>
    <span class="topbar-badge"><span class="badge-dot"></span> AI online</span>
    <span class="topbar-sep">Powered by ML · Auto-routing</span>
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
    "Server & Infrastructure Issue": ["server","database","outage","infrastructure","down","production","backend","hosting"],
    "Security Issue":                ["breach","hack","ransomware","malware","security","phishing","unauthorized","suspicious"],
    "Network & VPN Issue":           ["vpn","network","internet","firewall","wifi","bandwidth","latency","connectivity"],
    "Account & Access Issue":        ["login","password","account","access","reset","locked","credentials","sign in","2fa","mfa","authenticate"],
    "Software & App Issue":          ["app","software","crash","bug","error","update","patch","version","install"],
    "Hardware Issue":                ["hardware","printer","laptop","monitor","keyboard","mouse","device","screen","battery"],
    "Billing & Payment Issue":       ["billing","invoice","payment","refund","charge","subscription","overcharged"],
}

SOLUTION_MAP = {
    "Server & Infrastructure Issue": ["Restarted server cluster","Fixed database connection pool","Increased server capacity"],
    "Network & VPN Issue":           ["Reset VPN configuration","Checked and updated firewall rules","Rerouted traffic"],
    "Account & Access Issue":        ["Reset user credentials","Unlocked the account","Re-enabled 2FA enrollment"],
    "Security Issue":                ["Rotated all affected credentials","Blocked suspicious IPs","Escalated to Security Team"],
    "Software & App Issue":          ["Reinstalled the application","Applied latest security patch","Cleared cache and restarted"],
    "Hardware Issue":                ["Replaced faulty hardware unit","Updated device drivers","Dispatched IT field technician"],
    "Billing & Payment Issue":       ["Issued refund to original payment method","Corrected invoice details","Escalated to Finance Team"],
}

TEAM_MAP = {
    "Network & VPN Issue":           "Network Team",
    "Server & Infrastructure Issue": "Infrastructure Team",
    "Billing & Payment Issue":       "Finance Team",
    "Hardware Issue":                "IT Support Team",
    "Software & App Issue":          "Application Support Team",
    "Security Issue":                "Security Team",
    "Account & Access Issue":        "Identity Team",
}

PRIORITY_COLORS = {
    "Critical": "#ef4444",
    "High":     "#f59e0b",
    "Medium":   "#3b82f6",
    "Low":      "#22c55e",
}

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(BASE_DIR, "vectorizer.pkl"),      "rb") as f: vectorizer     = pickle.load(f)
    with open(os.path.join(BASE_DIR, "model.pkl"),           "rb") as f: model          = pickle.load(f)
    with open(os.path.join(BASE_DIR, "priority_model.pkl"),  "rb") as f: priority_model = pickle.load(f)
    with open(os.path.join(BASE_DIR, "priority_scaler.pkl"), "rb") as f: scaler         = pickle.load(f)
    return vectorizer, model, priority_model, scaler

vectorizer, model, priority_model, scaler = load_models()

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def assign_label(text: str) -> str:
    t = str(text).lower()
    for category, keywords in RULE_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return category
    return "General Support"


def build_priority_features(texts, categories, vectorizer, scaler):
    tfidf_feats = vectorizer.transform(texts)
    extra = []
    for text, cat in zip(texts, categories):
        t     = str(text).lower()
        words = t.split()
        length   = min(len(words) / 60.0, 1.0)
        severity = CATEGORY_SEVERITY.get(cat, 0.2)
        urgency  = sum(1 if w in t else 0 for w in ["urgent","critical","down","not working"]) / 4
        exclaim  = text.count("!") / max(len(text), 1)
        caps     = sum(1 for w in words if w.isupper()) / max(len(words), 1)
        extra.append([length, severity, urgency, exclaim, caps])
    extra_scaled = scaler.transform(np.array(extra))
    return hstack([tfidf_feats, extra_scaled])


def get_team(label: str, priority: str = None) -> str:
    base = TEAM_MAP.get(label, "Support Team")
    return f"Senior {base}" if priority == "Critical" else base


def extract_entities(text: str, category: str = "") -> dict:
    t = text.lower()

    module = (
        "Server"          if any(k in t for k in ["server","production","backend","hosting"]) else
        "Database"        if any(k in t for k in ["database","sql","mysql","postgres","db"]) else
        "VPN"             if any(k in t for k in ["vpn","network","wifi","internet","firewall"]) else
        "API"             if any(k in t for k in ["api","endpoint","webhook","rest"]) else
        "Application"     if any(k in t for k in ["application","software","app","program","tool"]) else
        "Auth Module"     if any(k in t for k in ["login","password","account","reset","credentials","sign in","2fa","mfa"]) else
        "Billing"         if any(k in t for k in ["billing","invoice","payment","refund","charge"]) else
        "Hardware"        if any(k in t for k in ["hardware","printer","laptop","monitor","keyboard","device"]) else
        "Auth Module"     if "Account"        in category else
        "Core System"     if "Infrastructure" in category else
        "Network Layer"   if "Network"        in category else
        "App Layer"       if "Software"       in category else
        "IT Hardware"     if "Hardware"       in category else
        "Finance System"  if "Billing"        in category else
        "Security Layer"  if "Security"       in category else
        "Support System"
    )

    platform = (
        "AWS"          if "aws"          in t else
        "Azure"        if "azure"        in t else
        "GCP"          if "gcp"          in t or "google cloud" in t else
        "Windows"      if "windows"      in t else
        "Linux"        if "linux"        in t else
        "macOS"        if "macos"        in t or "mac os" in t else
        "On-Premise"   if any(k in t for k in ["on-prem","on premise","local","datacenter"]) else
        "Web Browser"  if any(k in t for k in ["browser","chrome","firefox","safari","edge"]) else
        "Mobile"       if any(k in t for k in ["mobile","android","ios","iphone","phone"]) else
        "Cloud Infra"   if "Infrastructure" in category else
        "Internal VPN"  if "Network"        in category else
        "Corporate IT"  if "Hardware"       in category else
        "Web Platform"  if "Software"       in category else
        "Web Platform"  if "Account"        in category else
        "Cloud Platform"
    )

    issue = (
        "Failure"          if any(k in t for k in ["down","not working","failed","failure","unreachable","unavailable","offline"]) else
        "Crash"            if any(k in t for k in ["crash","crashed","stopped","freezing","hangs"]) else
        "Performance"      if any(k in t for k in ["slow","latency","lag","timeout","high load","unresponsive"]) else
        "Auth Failure"     if any(k in t for k in ["login","password","reset","credentials","locked","access denied","unable to"]) else
        "Error"            if any(k in t for k in ["error","bug","exception","invalid","broken"]) else
        "Access Block"     if any(k in t for k in ["cannot","can't","blocked","locked","restricted"]) else
        "Data Issue"       if any(k in t for k in ["missing","incorrect","wrong","corrupt","lost"]) else
        "Service Down"     if "Infrastructure" in category else
        "Auth Failure"     if "Account"        in category else
        "Connectivity"     if "Network"        in category else
        "App Malfunction"  if "Software"       in category else
        "Device Fault"     if "Hardware"       in category else
        "Payment Error"    if "Billing"        in category else
        "Security Breach"  if "Security"       in category else
        "General Fault"
    )

    return {"Module": module, "Platform": platform, "Issue": issue}


def generate_ticket_id() -> str:
    return f"TKT-#{random.randint(10000, 99999)}"


def predict_full(ticket: str) -> dict:
    vec         = vectorizer.transform([ticket])
    model_label = model.predict(vec)[0]
    probs       = model.predict_proba(vec)[0]
    confidence  = round(float(max(probs)) * 100, 1)

    rule_label = assign_label(ticket)
    category   = rule_label if rule_label != "General Support" else model_label

    if rule_label != "General Support":
        confidence = max(confidence, 82.0)

    X_p          = build_priority_features([ticket], [category], vectorizer, scaler)
    priority     = priority_model.predict(X_p)[0]
    team         = get_team(category, priority)
    routing_conf = round(confidence * 0.95, 1)
    solutions    = SOLUTION_MAP.get(category, ["Escalated to L2 Support"])
    entities     = extract_entities(ticket, category)
    decision     = "Auto Assigned" if confidence > 75 else "Human Review"
    ticket_id    = generate_ticket_id()
    timestamp    = datetime.now().strftime("%b %d %Y · %H:%M UTC")

    return {
        "ticket":       ticket,
        "category":     category,
        "priority":     priority,
        "team":         team,
        "confidence":   confidence,
        "routing_conf": routing_conf,
        "solutions":    solutions,
        "entities":     entities,
        "decision":     decision,
        "ticket_id":    ticket_id,
        "timestamp":    timestamp,
    }


# ─────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────
PRIORITY_ICON = {"Critical": "🔴", "High": "🟠", "Medium": "🔵", "Low": "🟢"}
PILL_CSS      = {"Critical": "pill pill-critical", "High": "pill pill-high",
                 "Medium": "pill pill-medium", "Low": "pill pill-low"}

def pill_html(p: str) -> str:
    return f'<span class="{PILL_CSS.get(p, "pill")}">{p.upper()}</span>'


def build_ai_text(r: dict) -> str:
    sols     = "\n".join(f"  • {s}" for s in r["solutions"])
    ent      = r["entities"]
    dec_icon = "✔" if r["decision"] == "Auto Assigned" else "⚠"
    pri_icon = PRIORITY_ICON.get(r["priority"], "🔵")
    return (
        f"🎫  TICKET ANALYSIS REPORT\n"
        f"{'─' * 44}\n\n"
        f"📌 Category      {r['category']}\n"
        f"🏢 Team          {r['team']}\n"
        f"{pri_icon} Priority       {r['priority'].upper()}\n\n"
        f"{'─' * 44}\n"
        f"📊 Confidence\n"
        f"   Category  →  {r['confidence']}%\n"
        f"   Routing   →  {r['routing_conf']}%\n\n"
        f"{'─' * 44}\n"
        f"⚙️  Entities\n"
        f"   Module    →  {ent['Module']}\n"
        f"   Platform  →  {ent['Platform']}\n"
        f"   Issue     →  {ent['Issue']}\n\n"
        f"{'─' * 44}\n"
        f"💡 Past Resolutions\n"
        f"{sols}\n\n"
        f"{'─' * 44}\n"
        f"{dec_icon}  Decision: {r['decision']}\n"
    )


def render_bubble_user(text: str):
    import html as _h
    st.markdown(
        f'<div class="bubble-user">{_h.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_bubble_ai(r: dict):
    import html as _h
    safe = _h.escape(build_ai_text(r))
    st.markdown(
        f'<div class="bubble-ai-row">'
        f'<div class="ai-avatar">🤖</div>'
        f'<div class="bubble-ai">{safe}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_bubble_thanks(r: dict):
    """Separate green thank-you bubble — always last in the conversation."""
    st.markdown(
        f"""<div class="bubble-ai-row" style="margin-top:2px">
              <div class="ai-avatar">🤖</div>
              <div class="bubble-thanks">
                <div style="font-size:20px;margin-bottom:6px">🙏</div>
                <div class="ty-head">Thank you for reaching out!</div>
                <div class="ty-body">
                  Your ticket has been <span class="ty-hi">logged &amp; assigned</span>
                  to the <span class="ty-hi">{r['team']}</span>.<br>
                  You'll receive a follow-up <span class="ty-hi">shortly</span>. We're on it! ✅
                  <div class="ticket-chip">{r['ticket_id']} &nbsp;·&nbsp; {r['decision']} &nbsp;·&nbsp; {r['timestamp']}</div>
                </div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def render_insights(r: dict):
    # Priority
    st.markdown(
        f'<div class="icard"><div class="icard-label">Priority</div>{pill_html(r["priority"])}</div>',
        unsafe_allow_html=True,
    )
    # Category
    st.markdown(
        f'<div class="icard"><div class="icard-label">Category</div>'
        f'<div class="icard-val">{r["category"]}</div></div>',
        unsafe_allow_html=True,
    )
    # Team
    st.markdown(
        f'<div class="icard"><div class="icard-label">Assigned Team</div>'
        f'<div class="icard-val">{r["team"]}</div></div>',
        unsafe_allow_html=True,
    )
    # Confidence bars
    col_c = PRIORITY_COLORS.get(r["priority"], "#3b82f6")
    st.markdown(
        f"""<div class="icard">
              <div class="icard-label">Confidence</div>
              <div class="conf-wrap">
                <div class="conf-row">
                  <span class="conf-label">Category</span>
                  <span class="conf-pct">{r['confidence']}%</span>
                </div>
                <div class="conf-track">
                  <div class="conf-fill" style="width:{int(r['confidence'])}%;background:{col_c}"></div>
                </div>
              </div>
              <div class="conf-wrap" style="margin-top:8px">
                <div class="conf-row">
                  <span class="conf-label">Routing</span>
                  <span class="conf-pct">{r['routing_conf']}%</span>
                </div>
                <div class="conf-track">
                  <div class="conf-fill" style="width:{int(r['routing_conf'])}%;background:#1f6feb"></div>
                </div>
              </div>
            </div>""",
        unsafe_allow_html=True,
    )
    # Entities
    ent = r["entities"]
    st.markdown(
        f"""<div class="icard">
              <div class="icard-label">Entities Detected</div>
              <table class="etable">
                <tr><td>Module</td><td>{ent['Module']}</td></tr>
                <tr><td>Platform</td><td>{ent['Platform']}</td></tr>
                <tr><td>Issue Type</td><td>{ent['Issue']}</td></tr>
              </table>
            </div>""",
        unsafe_allow_html=True,
    )
    # Solutions
    items = "".join(f'<li><span class="sol-dot">▸</span>{s}</li>' for s in r["solutions"])
    st.markdown(
        f'<div class="icard"><div class="icard-label">Past Resolutions</div>'
        f'<ul class="sol-list">{items}</ul></div>',
        unsafe_allow_html=True,
    )
    # Decision
    badge = (
        '<span class="dec-ok">✔&nbsp; Auto Assigned</span>'
        if r["decision"] == "Auto Assigned"
        else '<span class="dec-warn">⚠&nbsp; Human Review Required</span>'
    )
    st.markdown(
        f'<div class="icard"><div class="icard-label">Routing Decision</div>{badge}</div>',
        unsafe_allow_html=True,
    )
    # Ticket ID
    st.markdown(
        f'<div class="icard">'
        f'<div class="icard-label">Ticket ID</div>'
        f'<div class="icard-val" style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:#79c0ff">'
        f'{r["ticket_id"]}</div>'
        f'<div style="font-size:10px;color:#484f58;margin-top:3px">{r["timestamp"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
# roles: "user" | "assistant" | "thanks"
if "messages"    not in st.session_state: st.session_state.messages    = []
if "last_result" not in st.session_state: st.session_state.last_result = None

# ─────────────────────────────────────────────────────────────
# LAYOUT  —  LEFT: Conversation  |  RIGHT: Ticket Insights
# ─────────────────────────────────────────────────────────────
chat_col, info_col = st.columns([2, 1], gap="medium")

# ── LEFT: Conversation ───────────────────────────────────────
with chat_col:
    st.markdown('<div class="col-header">💬 Conversation</div>', unsafe_allow_html=True)

    # Welcome bubble when no history yet
    if not st.session_state.messages:
        st.markdown(
            """<div class="bubble-ai-row">
                 <div class="ai-avatar">🤖</div>
                 <div class="bubble-welcome">
                   Hello! Describe your support issue and I'll automatically
                   classify it, set priority, and route it to the right team.
                 </div>
               </div>""",
            unsafe_allow_html=True,
        )

    # Replay full conversation from session state
    for msg in st.session_state.messages:
        if   msg["role"] == "user":       render_bubble_user(msg["text"])
        elif msg["role"] == "assistant":  render_bubble_ai(msg["result"])
        elif msg["role"] == "thanks":     render_bubble_thanks(msg["result"])

    # Chat input
    user_input = st.chat_input("Describe your issue…")

    if user_input and user_input.strip():
        # 1. User bubble
        render_bubble_user(user_input)
        st.session_state.messages.append({"role": "user", "text": user_input, "result": None})

        # 2. Predict
        with st.spinner("Analyzing ticket…"):
            result = predict_full(user_input)
        st.session_state.last_result = result

        # 3. Analysis bubble
        render_bubble_ai(result)
        st.session_state.messages.append({"role": "assistant", "text": "", "result": result})

        # 4. Thank-you bubble  ← always last in conversation
        render_bubble_thanks(result)
        st.session_state.messages.append({"role": "thanks", "text": "", "result": result})

        st.rerun()

# ── RIGHT: Ticket Insights ───────────────────────────────────
with info_col:
    st.markdown('<div class="col-header">📊 Ticket Insights</div>', unsafe_allow_html=True)

    if st.session_state.last_result:
        render_insights(st.session_state.last_result)
    else:
        st.markdown(
            """<div class="empty-state">
                 <div class="empty-icon">📥</div>
                 Submit a ticket to see<br>AI-generated insights
               </div>""",
            unsafe_allow_html=True,
        )
