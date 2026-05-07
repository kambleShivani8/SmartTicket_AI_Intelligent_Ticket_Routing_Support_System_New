import numpy as np
import re
import os
import gdown
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
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
[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 0.5px solid #21262d !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary { font-size: 12px !important; color: #8b949e !important; }
[data-testid="stExpander"] p,
[data-testid="stExpander"] div     { font-size: 12px !important; color: #8b949e !important; }
[data-testid="stChatInput"] textarea {
    background: #161b22 !important;
    border: 0.5px solid #30363d !important;
    border-radius: 8px !important;
    color: #cdd9e5 !important;
    font-size: 13px !important;
}
[data-testid="stChatInput"] textarea:focus { border-color: #1f6feb !important; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
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
    "Security Issue":                0.95,
    "Network & VPN Issue":           0.8,
    "Account & Access Issue":        0.7,
    "Software & App Issue":          0.6,
    "Hardware Issue":                0.55,
    "Billing & Payment Issue":       0.5,
    "Setup & Configuration Issue":   0.4,
    "Shipping & Delivery Issue":     0.35,
    "Cancellation Request":          0.25,
    "General Support":               0.2,
}

RULE_KEYWORDS = {
    "Server & Infrastructure Issue": ["server", "database", "outage", "infrastructure", "down"],
    "Security Issue":                ["breach", "hack", "ransomware", "malware", "data leak", "security"],
    "Network & VPN Issue":           ["vpn", "network", "connectivity", "internet", "firewall"],
    "Account & Access Issue":        ["login", "password", "account", "access", "locked", "authentication"],
    "Software & App Issue":          ["app", "software", "crash", "bug", "error", "application"],
    "Hardware Issue":                ["hardware", "device", "printer", "laptop", "monitor", "keyboard"],
    "Billing & Payment Issue":       ["billing", "invoice", "payment", "charge", "refund"],
    "Setup & Configuration Issue":   ["setup", "configure", "install", "configuration"],
    "Shipping & Delivery Issue":     ["shipping", "delivery", "package", "courier", "tracking"],
    "Cancellation Request":          ["cancel", "cancellation", "terminate", "unsubscribe"],
}

SOLUTION_MAP = {
    "Server & Infrastructure Issue": ["Restarted server cluster", "Fixed database connection", "Increased server capacity"],
    "Network & VPN Issue":           ["Reset VPN config", "Checked firewall rules", "Restarted adapter"],
    "Account & Access Issue":        ["Reset credentials", "Unlocked account in AD", "Issued temp token"],
    "Security Issue":                ["Isolated affected system", "Rotated credentials", "Escalated to Sec team"],
    "Software & App Issue":          ["Cleared app cache", "Reinstalled software", "Applied latest patch"],
    "Hardware Issue":                ["Replaced faulty part", "Ran diagnostics", "Issued replacement device"],
    "Billing & Payment Issue":       ["Processed refund", "Corrected invoice", "Updated payment method"],
}

TEAM_MAP = {
    "Network & VPN Issue":           "Network Team",
    "Server & Infrastructure Issue": "Infrastructure Team",
    "Billing & Payment Issue":       "Finance Team",
    "Hardware Issue":                "IT Support Team",
    "Software & App Issue":          "Application Support Team",
    "Security Issue":                "Security Team",
    "Account & Access Issue":        "Identity Team",
    "Shipping & Delivery Issue":     "Logistics Team",
    "Setup & Configuration Issue":   "Onboarding Team",
    "Cancellation Request":          "Retention Team",
}

# ─────────────────────────────────────────────────────────────
# GOOGLE DRIVE DATASET LINK & FILE ID
# Drive link : https://drive.google.com/file/d/1Hahmz8xCREkriay4RDssA0jl2S3Bb48h/view?usp=drive_link
# ─────────────────────────────────────────────────────────────
GDRIVE_FILE_ID = "1Hahmz8xCREkriay4RDssA0jl2S3Bb48h"
DATA_PATH      = "data/combined_tickets.csv"


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def download_data():
    """Download dataset from Google Drive if not already present."""
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        url = "https://drive.google.com/uc?id=1Hahmz8xCREkriay4RDssA0jl2S3Bb48h"
        gdown.download(url, DATA_PATH, quiet=False)



def detect_and_rename(df):
    """
    Auto-detect text and label columns regardless of CSV column names.
    Supports: 'Ticket Description'/'Ticket Type', 'Document'/'Topic_group',
              'text'/'label', or any first two string columns.
    """
    df.columns = df.columns.str.strip()
    col_map = {}

    # known text column names
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["ticket description", "document", "text", "description", "body", "message", "content"]:
            col_map["text"] = c
            break

    # known label column names
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["ticket type", "topic_group", "label", "category", "class", "type"]:
            col_map["label"] = c
            break

    # fallback: pick first two string columns
    if "text" not in col_map or "label" not in col_map:
        str_cols = [c for c in df.columns if df[c].dtype == object]
        if len(str_cols) >= 2:
            if "text"  not in col_map: col_map["text"]  = str_cols[0]
            if "label" not in col_map: col_map["label"] = str_cols[1]

    df = df.rename(columns={col_map["text"]: "text", col_map["label"]: "label"})
    return df[["text", "label"]].dropna()


def fast_clean(texts):
    cleaned = []
    for t in texts:
        t = str(t).lower()
        t = re.sub(r'[^a-z\s]', ' ', t)
        t = re.sub(r'\s+', ' ', t).strip()
        cleaned.append(t)
    return cleaned


def assign_label(text):
    t = str(text).lower()
    for category, keywords in RULE_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return category
    return "General Support"


def assign_priority_label_smart(row):
    text     = str(row["text"]).lower()
    category = row.get("label", "General Support")
    keyword_score = 0.0
    if any(w in text for w in ["down", "outage", "critical", "breach", "hack", "ransomware", "data leak"]):
        keyword_score = 1.0
    elif any(w in text for w in ["fail", "error", "not working", "issue", "cannot", "unable"]):
        keyword_score = 0.65
    elif any(w in text for w in ["slow", "delay", "performance"]):
        keyword_score = 0.35
    severity_score = CATEGORY_SEVERITY.get(category, 0.2)
    length_score   = min(len(text.split()) / 60.0, 1.0)
    composite      = (0.5 * keyword_score) + (0.35 * severity_score) + (0.15 * length_score)
    if composite >= 0.75:  return "Critical"
    elif composite >= 0.5: return "High"
    elif composite >= 0.3: return "Medium"
    else:                  return "Low"


def build_priority_features(texts, categories, vectorizer, scaler=None, fit=False):
    tfidf_feats = vectorizer.transform(texts)
    extra = []
    for text, cat in zip(texts, categories):
        t     = str(text).lower()
        words = t.split()
        length        = min(len(words) / 60.0, 1.0)
        severity      = CATEGORY_SEVERITY.get(cat, 0.2)
        urgency_kws   = ["urgent", "asap", "critical", "immediately", "emergency",
                         "down", "broken", "not working", "cannot", "production"]
        urgency_count = sum(1 for w in urgency_kws if w in t) / len(urgency_kws)
        exclamation   = text.count("!") / max(len(text), 1)
        caps_ratio    = sum(1 for w in words if w.isupper()) / max(len(words), 1)
        extra.append([length, severity, urgency_count, exclamation, caps_ratio])
    extra = np.array(extra)
    if fit:
        scaler       = MinMaxScaler()
        extra_scaled = scaler.fit_transform(extra)
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be passed when fit=False.")
        extra_scaled = scaler.transform(extra)
    return hstack([tfidf_feats, extra_scaled]), scaler


def get_team(label, priority=None):
    base = TEAM_MAP.get(label, "Support Team")
    return f"Senior {base}" if priority == "Critical" else base


def extract_entities(text):
    t = text.lower()
    module_map   = {"server": "Server", "database": "Database", "api": "API",
                    "application": "Application", "vpn": "VPN", "network": "Network"}
    platform_map = {"aws": "AWS", "azure": "Azure", "windows": "Windows", "linux": "Linux"}
    module   = next((v for k, v in module_map.items()   if k in t), None)
    platform = next((v for k, v in platform_map.items() if k in t), None)
    if "crash" in t:                        issue = "Crash"
    elif "slow" in t:                       issue = "Performance"
    elif "error" in t:                      issue = "Error"
    elif "down" in t or "not working" in t: issue = "Failure"
    else:                                   issue = None
    return {"Module": module, "Platform": platform, "Issue": issue}


# ─────────────────────────────────────────────────────────────
# LOAD & TRAIN  (cached — runs once per session)
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_and_train():
    # ── 1. Download dataset from Google Drive ──
    download_data()

    # ── 2. Load & normalise columns ──
    final_df = pd.read_csv(DATA_PATH)
    final_df = detect_and_rename(final_df)
    final_df["text"] = final_df["text"].fillna("")

    # ── 3. Sample to stay within Streamlit Cloud memory (≤ 1 GB) ──
    if len(final_df) > 15000:
        final_df = final_df.sample(15000, random_state=42).reset_index(drop=True)

    final_df["clean_text"] = fast_clean(final_df["text"])

    # ── 4. TF-IDF vectoriser ──
    vectorizer = TfidfVectorizer(
        max_features=3000, ngram_range=(1, 1),
        min_df=2, max_df=0.9, stop_words="english"
    )
    vectorizer.fit(final_df["clean_text"])

    # ── 5. Category model ──
    X_cat = vectorizer.transform(final_df["clean_text"])
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_cat, final_df["label"], test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    model.fit(X_tr_c, y_tr_c)

    # ── 6. Priority labels & model ──
    final_df["priority"] = final_df.apply(assign_priority_label_smart, axis=1)

    X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
        final_df["text"], final_df["priority"], final_df["label"],
        test_size=0.2, random_state=42
    )
    X_train_vec, scaler = build_priority_features(X_train, cat_train, vectorizer, fit=True)
    X_test_vec,  _      = build_priority_features(X_test,  cat_test,  vectorizer, scaler=scaler)
    priority_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    priority_model.fit(X_train_vec, y_train)

    # ── 7. SBERT — encode only a 2000-row sample to save RAM ──
    sbert     = SentenceTransformer("all-MiniLM-L6-v2")
    sbert_df  = final_df.sample(min(2000, len(final_df)), random_state=42).reset_index(drop=True)
    ticket_embeddings = sbert.encode(
        sbert_df["text"].tolist(), show_progress_bar=False, batch_size=64
    )

    return vectorizer, model, priority_model, scaler, sbert, ticket_embeddings, sbert_df


vectorizer, model, priority_model, scaler, sbert, ticket_embeddings, final_df = load_and_train()


# ─────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────
def predict_full(ticket: str) -> dict:
    vec = vectorizer.transform([ticket])

    model_label = model.predict(vec)[0]
    probs       = model.predict_proba(vec)[0]
    confidence  = round(float(max(probs)) * 100, 1)

    rule_label = assign_label(ticket)
    category   = rule_label if rule_label != "General Support" else model_label

    X_p, _   = build_priority_features([ticket], [category], vectorizer, scaler=scaler)
    priority  = priority_model.predict(X_p)[0]

    team         = get_team(category, priority)
    routing_conf = round(confidence * 0.95, 1)

    emb     = sbert.encode([ticket])
    sim     = cosine_similarity(emb, ticket_embeddings)[0]
    top_idx = np.argsort(sim)[-3:][::-1]
    similar = [final_df.iloc[i]["text"] for i in top_idx]

    solutions = SOLUTION_MAP.get(category, ["No known fix"])
    entities  = extract_entities(ticket)
    decision  = "Auto Assigned" if confidence > 75 else "Human Review"

    return dict(
        category=category, priority=priority, team=team,
        confidence=confidence, routing_conf=routing_conf,
        similar=similar, solutions=solutions,
        entities=entities, decision=decision,
    )


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────
PILL_CSS = {
    "Critical": "pill pill-critical",
    "High":     "pill pill-high",
    "Medium":   "pill pill-medium",
    "Low":      "pill pill-low",
}

def priority_pill(p: str) -> str:
    cls = PILL_CSS.get(p, "pill")
    return f'<span class="{cls}">{p.upper()}</span>'


def render_ai_bubble(r: dict) -> str:
    dec_cls  = "decision-ok"   if r["decision"] == "Auto Assigned" else "decision-warn"
    dec_icon = "✔"             if r["decision"] == "Auto Assigned" else "⚠"
    dec_text = "Auto Assigned (confidence > 75%)" if r["decision"] == "Auto Assigned" \
               else "Flagged for Human Review"
    return f"""
<div class="bubble-ai-wrap">
  <div class="ai-avatar">🤖</div>
  <div class="bubble-ai">
    <div class="ai-field"><span class="ai-label">Category</span>
      <span class="ai-val" style="font-size:12px;">{r['category']}</span></div>
    <div class="ai-field"><span class="ai-label">Priority</span>
      {priority_pill(r['priority'])}</div>
    <div class="ai-field"><span class="ai-label">Team</span>
      <span class="ai-val" style="font-size:12px;">{r['team']}</span></div>
    <div class="ai-field"><span class="ai-label">Confidence</span>
      <span class="ai-val">{r['confidence']}%</span></div>
    <div class="{dec_cls}">{dec_icon} {dec_text}</div>
  </div>
</div>"""


def render_insights(r: dict):
    conf_w = int(r["confidence"])

    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Priority</div>
      {priority_pill(r['priority'])}
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Category</div>
      <div class="icard-val" style="font-size:12px;">{r['category']}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Assigned team</div>
      <div class="icard-val" style="font-size:12px;">{r['team']}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Confidence</div>
      <div class="icard-val">{r['confidence']}%</div>
      <div class="conf-track"><div class="conf-fill" style="width:{conf_w}%"></div></div>
    </div>""", unsafe_allow_html=True)

    ent = r["entities"]
    st.markdown(f"""
    <div class="icard">
      <div class="icard-label">Entities</div>
      <table class="etable">
        <tr><td>Module</td>  <td>{ent.get('Module')   or 'N/A'}</td></tr>
        <tr><td>Platform</td><td>{ent.get('Platform') or 'N/A'}</td></tr>
        <tr><td>Issue</td>   <td>{ent.get('Issue')    or 'N/A'}</td></tr>
      </table>
    </div>""", unsafe_allow_html=True)

    with st.expander("🧠 Similar tickets"):
        for t in r["similar"]:
            snippet = t[:140] + ("…" if len(t) > 140 else "")
            st.write(f"– {snippet}")

    with st.expander("💡 Suggested solutions"):
        for s in r["solutions"]:
            st.write(f"– {s}")

    dec_cls  = "decision-ok"   if r["decision"] == "Auto Assigned" else "decision-warn"
    dec_icon = "✔"             if r["decision"] == "Auto Assigned" else "⚠"
    st.markdown(
        f'<div class="{dec_cls}" style="font-size:12px;margin-top:4px;">'
        f'{dec_icon} {r["decision"]} · Routing {r["routing_conf"]}%</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "messages"    not in st.session_state: st.session_state.messages    = []
if "last_result" not in st.session_state: st.session_state.last_result = None

# ─────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────
chat_col, info_col = st.columns([2, 1], gap="medium")

# ── LEFT: CHAT ──
with chat_col:
    st.markdown('<div class="col-header">💬 Conversation</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="bubble-ai-wrap">
          <div class="ai-avatar">🤖</div>
          <div class="bubble-ai">
            Hello! Describe your support issue and I'll classify it,
            assign a priority, and route it to the right team.
          </div>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(msg["content"], unsafe_allow_html=True)

    user_input = st.chat_input("Describe your issue…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="bubble-user">{user_input}</div>', unsafe_allow_html=True)

        with st.spinner("Analyzing ticket…"):
            result = predict_full(user_input)

        st.session_state.last_result = result
        ai_html = render_ai_bubble(result)
        st.markdown(ai_html, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": ai_html})

# ── RIGHT: INSIGHTS ──
with info_col:
    st.markdown('<div class="col-header">📊 Ticket insights</div>', unsafe_allow_html=True)

    if st.session_state.last_result:
        render_insights(st.session_state.last_result)
    else:
        st.markdown("""
        <div style="text-align:center;color:#484f58;font-size:13px;padding:40px 0;">
          <div style="font-size:28px;margin-bottom:8px;">📥</div>
          Enter a ticket to see insights
        </div>""", unsafe_allow_html=True)
