# 🏦 CrestBank AI Personal Finance Advisor

An **Agentic AI Personal Finance Advisor** built with **LangGraph**, **Google Gemini (`google-genai`)**, and **Pydantic**, designed to upgrade a bank's chatbot from static monthly statements and passive transaction notifications into an **autonomous, multi-agent financial advisory system**.

---

## ✨ Key Capabilities Demonstrated

1. **Multi-Agent Architecture (`src/agents/`)**:
   - **Planner Agent (`planner_agent.py`)**: Understands user intent, consults Short-Term & Long-Term Memory, constructs an explicit `ExecutionPlan`, coordinates specialized sub-agents, and synthesizes the final response.
   - **Spending Analysis Agent (`spending_agent.py`)**: Reads multi-month transaction data, categorizes spending (`Food & Dining`, `Groceries`, `Entertainment`, `Subscriptions`, `Shopping`, `Utilities`, `Transportation`), computes Month-over-Month (MoM) shifts, and detects recurring subscriptions & high-frequency micro-spend habits.
   - **Recommendation Agent (`recommendation_agent.py`)**: **Strictly consumes the structured `SpendingAnalysisReport` from the Spending Analysis Agent** + customer financial goals to simulate savings scenarios and budget optimizations.
   - **Alerting Agent (`alert_agent.py`)**: Runs **proactively** on session startup (upgrading passive bank notifications) and **reactively** during chat to flag budget breaches, statistical $Z$-score spending spikes, subscription creep, and cash-flow burn risks.
2. **LangGraph State Orchestration (`src/agents/orchestrator.py`)**:
   - Uses a compiled `StateGraph(FinanceAdvisorState)` with conditional edges that enforce the mandatory data pipeline: `Planner (Plan)` $\rightarrow$ `Spending Analysis Agent` $\rightarrow$ `Recommendation Agent` $\rightarrow$ `Alerting Agent` $\rightarrow$ `Planner (Synthesize)`.
3. **Deterministic Financial Tool Calling (`src/tools/`)**:
   - Zero LLM math hallucinations—all category aggregations, MoM deltas, $Z$-score anomaly checks, and compound savings projections execute in deterministic Python tools.
4. **Dual-Tier Memory Handling (`src/memory/memory_manager.py`)**:
   - **Short-Term Memory (STM)**: Sliding conversational context window + cached session analysis reports.
   - **Long-Term Memory (LTM)**: Persistent customer profile, financial goals, custom category budgets, and learned user constraints (e.g., remembering not to cancel essential work software or updating category budgets dynamically from chat).
5. **Dual-Mode Gemini Engine (`src/llm/engine.py`)**:
   - Supports live **Google Gemini (`gemini-2.5-flash`)** via free-tier Google AI Studio API key, **and** includes a built-in **Deterministic Financial Reasoning Fallback** so evaluators can run the entire system 100% offline out-of-the-box without an API key.

---

## 📂 Modular Project Structure

```text
Thougthscrest/
├── README.md                          # Setup, usage, and evaluation guide
├── ARCHITECTURE_AND_TRADEOFFS.md      # Detailed Architecture Decision Records (ADRs) & trade-offs
├── requirements.txt                   # Project dependencies (langgraph, google-genai, pydantic, streamlit, flask)
├── .env.example                       # Optional GOOGLE_API_KEY configuration template
├── app.py                             # Interactive Streamlit Conversational UI (clean customer view)
├── cli.py                             # Terminal CLI & Automated 3-Scenario Demo Walkthrough
├── logs/                              # Automatically generated LangGraph execution traces & audit logs
│   ├── agent_traces.jsonl             # Machine-readable JSONL trace of every agent step & handoff
│   └── CUST-001_traces.log            # Human-readable step-by-step agent execution trace per customer
├── src/
│   ├── config.py                      # Central configuration & thresholds
│   ├── schemas/
│   │   └── models.py                  # Pydantic V2 contracts & LangGraph FinanceAdvisorState TypedDict
│   ├── data/
│   │   ├── mock_generator.py          # Generates 3 customer personas x 3 months of realistic transactions
│   │   └── db.py                      # SQLite database access & seeding layer
│   ├── tools/
│   │   ├── transaction_tools.py       # Category aggregation, MoM comparison, subscription & micro-spend detection
│   │   ├── anomaly_tools.py           # Budget utilization, Z-score outlier detection, cash-flow burn checks
│   │   └── simulation_tools.py        # Quantified savings & budget optimization simulator
│   ├── memory/
│   │   └── memory_manager.py          # Dual-tier Short-Term + Long-Term memory store
│   ├── llm/
│   │   └── engine.py                  # Google Gemini adapter with automatic offline fallback
│   └── agents/
│       ├── planner_agent.py           # Agent 1: Planner & Synthesizer
│       ├── spending_agent.py          # Agent 2: Spending Analysis Agent
│       ├── recommendation_agent.py    # Agent 3: Recommendation Agent (consumes SpendingAnalysisReport)
│       ├── alerting_agent.py          # Agent 4: Proactive & Reactive Alerting Agent
│       └── orchestrator.py            # LangGraph StateGraph wiring & conditional routing
└── tests/
    └── test_agent_pipeline.py         # End-to-end unit & integration test suite
```

---

## 🚀 Quickstart & How to Run

### 1. Environment & Dependencies
Using Python 3.11+ (on macOS with Python 3.11 where `langgraph`, `google-genai`, `pydantic`, and `flask` are installed):
```bash
pip install -r requirements.txt
```
*(Optional)* To enable live Gemini generation, copy `.env.example` to `.env` and add your free [Google AI Studio API Key](https://aistudio.google.com/apikey), or paste it directly into the Web UI sidebar:
```bash
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY=your_key_here
```

### 2. Run the Automated 3-Persona Demo Walkthrough (CLI)
Runs 3 comprehensive evaluation scenarios across all customer personas, printing proactive alerts, the step-by-step **LangGraph execution trace**, state handoffs, and final markdown responses:
```bash
python3 cli.py --demo
```
*(Or run interactively in the terminal with `python3 cli.py`)*

### 3. Launch the Streamlit Conversational Web UI (`app.py`)
```bash
source .venv/bin/activate
streamlit run app.py
```
> **Trace Logging for Debugging & Submission Review**: To keep the customer chatbot interface clean, internal agent traces are not shown in the customer chat bubbles. Instead, every LangGraph execution turn is automatically logged to the `logs/` folder (`logs/agent_traces.jsonl` and `logs/<CUSTOMER_ID>_traces.log`).

### 4. Run the Automated Test Suite
Verifies mock data seeding, tool accuracy, the strict `Spending Analysis Agent -> Recommendation Agent` input contract, proactive alerting, and Long-Term Memory persistence:
```bash
python3 -m unittest discover -s tests -v
```

---

## 👥 Mock Customer Personas Included

| Customer ID | Name & Persona | Monthly Income | Planted Patterns for Agents to Discover |
| :--- | :--- | :---: | :--- |
| **`CUST-001`** | **Alex Rivera**<br>*Young Tech Professional* | \$6,200 | • **Food Delivery Surge**: `+72.6%` MoM jump (\$742 vs \$450 budget) from 11 DoorDash/UberEats orders.<br>• **Subscription Creep**: 6 recurring subscriptions (\$136.94/mo) with overlapping video & music apps.<br>• **Entertainment Breach**: \$335 spend vs \$250 budget (`134%`). |
| **`CUST-002`** | **Priya Sharma**<br>*Family Household Manager* | \$8,500 | • **Statistical Outlier Spike**: \$680 single transaction at HomeDepot Appliance Center ($Z\text{-score} > 2.2$).<br>• **Utility Bill Surge**: `+60.3%` MoM jump in electricity/gas.<br>• **Idle Cash Opportunity**: \$14,200 sitting in checking (`0.01%` APY) ready for High-Yield Vault sweep. |
| **`CUST-003`** | **Marcus Vance**<br>*Freelance Designer* | \$3,600–\$4,500 | • **Cash-Flow Burn Risk**: October income dip + \$890 checking buffer triggers `CRITICAL` liquidity alert.<br>• **Micro-Spend Drain**: 10x specialty coffee visits (\$115) + 8x rideshares (\$276) = \$391/mo.<br>• **Memory-Aware Protection**: LTM protects essential work tools (`Adobe CC`, `Figma`) from cancellation. |

---

## 📐 Architecture & Trade-Offs Documentation
See [ARCHITECTURE_AND_TRADEOFFS.md](file:///Users/vizzdd/Documents/FDE/Thougthscrest/ARCHITECTURE_AND_TRADEOFFS.md) for the complete system design, sequence diagrams, and Architectural Decision Records (ADRs) prepared for submission evaluation.

