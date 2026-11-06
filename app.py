"""
Streamlit Conversational UI for the Agentic AI Personal Finance Advisor (`streamlit run app.py`).
Showcases:
- Customer Persona Switcher (3 realistic banking personas)
- Proactive Alerting Agent Banner on session startup (replacing legacy static notifications)
- Interactive Multi-Turn Chatbot powered by LangGraph + Google Gemini
- Live "Under the Hood: Multi-Agent Execution Trace" inspector showing:
  1. Planner Agent's ExecutionPlan
  2. Spending Analysis Agent's `SpendingAnalysisReport` & Tool Calls
  3. Direct State Handoff (`SpendingAnalysisReport` -> `Recommendation Agent`)
  4. Alerting Agent's anomaly & budget breach scan
  5. Dual-Tier Memory (Short-Term & Long-Term Memory) inspector
"""

import streamlit as st
from src.agents.orchestrator import get_proactive_session_alerts, run_advisor_turn
from src.data.db import db
from src.llm.engine import llm_engine
from src.memory.memory_manager import memory_manager

st.set_page_config(
    page_title="CrestBank AI Personal Finance Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for banking dashboard feel
st.markdown(
    """
    <style>
    .alert-critical {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
        color: #881337;
    }
    .alert-warning {
        background-color: #fffbeb;
        border-left: 5px solid #d97706;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
        color: #78350f;
    }
    .alert-info {
        background-color: #eff6ff;
        border-left: 5px solid #2563eb;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 8px;
        color: #1e3a8a;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar: Persona Switcher, Gemini Key, and Memory Inspector ---
with st.sidebar:
    st.title("🏦 CrestBank AI")
    st.caption("Multi-Agent Personal Finance Advisor (LangGraph + Gemini)")

    profiles = db.list_profiles()
    profile_options = {p.customer_id: f"{p.name} ({p.customer_id})" for p in profiles}

    selected_cid = st.selectbox(
        "👤 Select Customer Persona",
        options=list(profile_options.keys()),
        format_func=lambda cid: profile_options[cid],
    )

    active_profile = db.get_profile(selected_cid)
    st.info(f"**Persona Profile:**\n{active_profile.persona_tag}")

    col1, col2 = st.columns(2)
    col1.metric("Net Income/mo", f"${active_profile.monthly_net_income:,.0f}")
    col2.metric("Checking Bal", f"${active_profile.checking_balance:,.0f}")

    st.markdown("**🎯 Active Financial Goals:**")
    for g in active_profile.financial_goals:
        pct = min(100, int((g.current_amount / g.target_amount) * 100))
        st.progress(pct / 100.0, text=f"{g.name}: ${g.current_amount:,.0f} / ${g.target_amount:,.0f} ({pct}%)")

    st.divider()
    st.subheader("🔑 Gemini LLM Settings")
    api_key_input = st.text_input(
        "Google Gemini API Key (optional)",
        value=llm_engine.api_key,
        type="password",
        help="Paste a free Gemini API key from Google AI Studio. If blank, runs using the built-in Deterministic Financial Reasoning Engine.",
    )
    if api_key_input != llm_engine.api_key:
        llm_engine.set_api_key(api_key_input)

    if llm_engine.is_live_gemini_enabled():
        st.success(f"Live Gemini Active (`{llm_engine.model_name}`)")
    else:
        st.warning("Mode: Hybrid Deterministic Fallback (Set GOOGLE_API_KEY for live Gemini)")

    st.divider()
    st.subheader("🧠 Long-Term Memory Store")
    ltm_data = memory_manager.get_long_term_memory(selected_cid)
    for fact in ltm_data.get("learned_facts", []):
        st.caption(f"• {fact}")

    if st.button("🔄 Reset Session & Re-seed Mock Data", use_container_width=True):
        db.seed_mock_data()
        memory_manager.clear_session(selected_cid)
        if "messages_by_cust" in st.session_state:
            st.session_state.messages_by_cust[selected_cid] = []
        st.rerun()

# --- Main Area Header ---
st.title(f"Personal Finance Advisor — {active_profile.name}")
st.caption(
    "Upgrading static statements & transaction notifications into autonomous multi-agent financial guidance."
)

# --- Proactive Alerting Agent Banner ---
proactive_alerts = get_proactive_session_alerts(selected_cid)
with st.expander(
    f"🔔 Proactive Alerting Agent Scan — {len(proactive_alerts.alerts)} Active Signals "
    f"(Overall Risk: {proactive_alerts.overall_risk_level})",
    expanded=True,
):
    for a in proactive_alerts.alerts:
        css_class = (
            "alert-critical"
            if a.severity == "CRITICAL"
            else ("alert-warning" if a.severity == "WARNING" else "alert-info")
        )
        st.markdown(
            f"""
            <div class="{css_class}">
                <strong>[{a.severity} • {a.alert_type}] {a.title}</strong> ({a.metric_highlight})<br/>
                {a.description}<br/>
                <em>Recommended Immediate Action: {a.recommended_immediate_action}</em>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- Quick Sample Prompts ---
st.markdown("**💬 Try an Autonomous Multi-Agent Prompt:**")
sample_cols = st.columns(4)
prompt_triggered = None

sample_prompts = [
    "Analyze my spending patterns this month and give me a personalized plan to save more.",
    "Why did my Food & Dining and Subscriptions jump so much? How do I optimize my budget?",
    "Are there any unusual spending spikes, budget breaches, or cashflow alerts on my account?",
    "I'm saving for a family trip in August—update my Food budget to $400 and show me how fast I reach my goals.",
]

for idx, col in enumerate(sample_cols):
    if col.button(sample_prompts[idx], key=f"sample_{idx}", use_container_width=True):
        prompt_triggered = sample_prompts[idx]

st.divider()

# Initialize session state for chat history per customer & track active customer switches
if "messages_by_cust" not in st.session_state:
    st.session_state.messages_by_cust = {
        "CUST-001": [],
        "CUST-002": [],
        "CUST-003": [],
    }
if "prev_selected_cid" not in st.session_state:
    st.session_state.prev_selected_cid = selected_cid
elif st.session_state.prev_selected_cid != selected_cid:
    st.session_state.prev_selected_cid = selected_cid

if selected_cid not in st.session_state.messages_by_cust:
    st.session_state.messages_by_cust[selected_cid] = []

st.caption(
    f"🔒 **Isolated Session Active for {active_profile.name} (`{selected_cid}`)** — "
    f"Chat history, Short-Term Memory, and Long-Term Memory are strictly scoped to `{selected_cid}`."
)

# Use a customer-scoped placeholder so switching personas cleanly replaces the entire chat view
chat_placeholder = st.empty()
with chat_placeholder.container():
    cust_messages = st.session_state.messages_by_cust[selected_cid]
    if not cust_messages:
        st.info(
            f"👋 Welcome, **{active_profile.name}**! Your isolated financial advisory session is ready. "
            f"Click one of the sample prompts above or ask a question below."
        )
    for msg_idx, msg in enumerate(cust_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Handle new chat input
chat_input_val = st.chat_input("Ask about your spending, budget optimization, subscriptions, or savings goals...")
active_query = prompt_triggered or chat_input_val

if active_query:
    with st.spinner(f"LangGraph Orchestrator running Planner -> Spending Analysis -> Recommendation -> Alerting Agents for {active_profile.name}..."):
        result_state = run_advisor_turn(selected_cid, active_query)
        final_reply = result_state["final_response"]

    st.session_state.messages_by_cust[selected_cid].append({"role": "user", "content": active_query})
    st.session_state.messages_by_cust[selected_cid].append(
        {
            "role": "assistant",
            "content": final_reply,
            "trace_state": result_state,
        }
    )
    st.rerun()

