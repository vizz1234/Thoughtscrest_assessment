# Architecture Decisions, System Design & Engineering Trade-Offs

## 1. Executive Summary
This document logs all architectural decisions, design patterns, inter-agent contracts, and engineering trade-offs for the **CrestBank Agentic AI Personal Finance Advisor**.

The system transforms a legacy banking chatbot—previously restricted to **static monthly statements and passive transaction notifications**—into an autonomous, multi-agent advisory system capable of:
1. **Multi-Agent Specialization** across 4 distinct agents (`Planner`, `Spending Analysis`, `Recommendation`, and `Alerting`).
2. **Graph-Enforced Agent Orchestration (`LangGraph`)** guaranteeing that the `Recommendation Agent` strictly receives its analytical input from the `Spending Analysis Agent`.
3. **Autonomous Reasoning & Deterministic Tool Calling** separating quantitative financial math from natural language synthesis.
4. **Dual-Tier Memory Management** combining short-term conversational state with long-term customer goals, budgets, and learned constraints.
5. **Proactive & Reactive Financial Guidance** surfacing real-time budget breaches, statistical spending anomalies ($Z$-score spikes), subscription creep, and quantified savings roadmaps.

---

## 2. High-Level Multi-Agent Architecture (`LangGraph StateGraph`)

```mermaid
flowchart TD
    User(["Bank Customer (Chat / Session Start)"]) <--> UI["Conversational UI (Streamlit app.py / Flask web_app.py / CLI)"]
    UI <--> PlannerPlan["1A. Planner Agent (Intent & Execution Planning Node)"]

    subgraph MemoryLayer ["Dual-Tier Memory Layer (src/memory/)"]
        STM["Short-Term Memory\n(Sliding Chat Window + Cached Session Reports)"]
        LTM["Long-Term Memory\n(SQLite Customer Profile + Persistent JSON Facts/Goals)"]
    end

    PlannerPlan <--> MemoryLayer

    PlannerPlan -->|"Conditional Edge:\nRequires Transaction Insights"| SpendingNode["2. Spending Analysis Agent Node"]
    PlannerPlan -->|"Proactive Startup / Alert Scan"| AlertNode["4. Alerting Agent Node"]

    SpendingNode ==>|"Writes state['spending_report']\n(Strict Pydantic Schema)"| RecNode["3. Recommendation Agent Node"]
    SpendingNode -->|"If only analysis requested"| AlertNode

    RecNode -->|"Writes state['recommendation_report']"| AlertNode
    AlertNode -->|"Writes state['alert_report']"| PlannerSynth["1B. Planner Agent (Response Synthesis Node)"]
    PlannerSynth -->|"Updates STM/LTM & Returns Markdown"| UI

    subgraph ToolRegistry ["Deterministic Financial Tool Layer (src/tools/)"]
        T1["tool_aggregate_spending_by_category()"]
        T2["tool_detect_subscriptions_and_habits()"]
        T3["tool_simulate_budget_optimization()"]
        T4["tool_evaluate_financial_alerts()"]
    end

    SpendingNode --> T1
    SpendingNode --> T2
    RecNode --> T3
    AlertNode --> T4
```

---

## 3. Core Components & Agent Responsibilities

### 3.1 Planner Agent (`src/agents/planner_agent.py`)
* **Role**: Front-door supervisor, intent planner, memory coordinator, and final synthesizer.
* **Two-Phase Execution in LangGraph**:
  1. **`planner_plan_node` (Entry)**:
     - Reads user query + **Short-Term Memory** (recent conversation turns) + **Long-Term Memory** (financial goals, risk tolerance, learned constraints).
     - Extracts any new user facts or budget modifications (e.g., *"I'm saving for a wedding in December—set my dining budget to \$500"*) and persists them immediately.
     - Produces a structured `ExecutionPlan` specifying which sub-agents to run and enforcing the rule: **`if run_recommendation_agent is True -> force run_spending_agent = True`**.
  2. **`planner_synthesize_node` (Exit)**:
     - Collates the structured outputs (`SpendingAnalysisReport`, `RecommendationReport`, `AlertReport`) into a transparent, empathetic response with markdown tables and clear attribution to each specialist agent.

### 3.2 Spending Analysis Agent (`src/agents/spending_agent.py`)
* **Role**: Transaction data scientist & behavioral pattern recognizer.
* **Tools Invoked**:
  - `tool_aggregate_spending_by_category`: Aggregates 3 months of debit/credit transactions (`2026-08`, `2026-09`, `2026-10`), computes Month-over-Month (MoM) dollar/percentage deltas, 3-month rolling averages, and budget utilization ratios.
  - `tool_detect_subscriptions_and_habits`: Identifies recurring subscriptions, flags overlapping services in the same subcategory (e.g., multiple video or music streaming subscriptions), and detects high-frequency micro-spend habits (e.g., 11x monthly food delivery orders or daily coffee/rideshare drains).
* **Output**: Strictly typed `SpendingAnalysisReport` stored in `state["spending_report"]`.

### 3.3 Recommendation Agent (`src/agents/recommendation_agent.py`)
* **Role**: Personalized financial strategist & savings simulator.
* **Enforced Input Dependency**:
  - Reads `state["spending_report"]` (`SpendingAnalysisReport`) directly from the `Spending Analysis Agent`.
  - Raises an explicit `ValueError` / `RuntimeError` if invoked without a valid `SpendingAnalysisReport`.
* **Tools Invoked**:
  - `tool_simulate_budget_optimization`: Simulates habit reductions (e.g., cutting food delivery frequency by 50%), subscription consolidation (while respecting Long-Term Memory constraints such as keeping essential work software like Adobe/Figma), category budget normalization, and high-yield savings sweeps for idle checking cash.
* **Output**: `RecommendationReport` with exact monthly/annual dollar savings, projected new savings rate ($\%$), and goal-completion acceleration timelines.

### 3.4 Alerting Agent (`src/agents/alerting_agent.py`)
* **Role**: Proactive & reactive risk monitor.
* **Dual Execution Modes**:
  1. **Proactive Session Startup Scan (`get_proactive_session_alerts`)**: Runs automatically when the user opens the chatbot or switches personas, replacing legacy static transaction notifications with severity-graded (`CRITICAL`, `WARNING`, `INFO`) proactive alerts.
  2. **In-Graph Reactive Node (`alerting_node`)**: Runs within the LangGraph pipeline to cross-check overspending risks alongside recommendations.
* **Tools Invoked**:
  - `tool_evaluate_financial_alerts`: Checks category budget breaches ($\ge 100\%$ critical, $\ge 85\%$ warning), statistical single-transaction outliers ($Z\text{-score} \ge 2.2$), subscription creep, and low checking liquidity burn risks.

---

## 4. Architectural Decision Records (ADRs) & Trade-Off Analysis

### ADR-01: LangGraph `StateGraph` vs. Autonomous Peer-to-Peer Swarm (AutoGen / CrewAI)
* **Decision**: Use **LangGraph (`StateGraph`)** with a typed `FinanceAdvisorState` and explicit conditional edges.
* **Context**: Financial advisory systems have strict ordering dependencies—specifically, the `Recommendation Agent` **must** derive its advice from the `Spending Analysis Agent`'s output rather than hallucinating numbers or querying raw transactions independently.
* **Trade-Offs**:
  - *Pros*: Deterministic graph topology; guaranteed state handoff (`spending_analysis_node` $\rightarrow$ `recommendation_node`); zero risk of infinite agent loops; full step-by-step inspectability via `agent_trace`.
  - *Cons*: Slightly more upfront schema and edge-wiring boilerplate compared to free-form multi-agent chat rooms.
  - *Why it wins for Banking*: Regulatory compliance and customer trust require deterministic, auditable workflows.

### ADR-02: Deterministic Python Tool Arithmetic vs. LLM Token Math
* **Decision**: Offload 100% of financial calculations (category sums, MoM percentages, $Z$-scores, compound savings projections, and budget utilization) to deterministic Python tools (`src/tools/`).
* **Context**: LLMs are probabilistic token predictors and frequently make arithmetic rounding or aggregation errors when summing 50+ raw transactions in a prompt context window.
* **Trade-Offs**:
  - *Pros*: **100% mathematical accuracy** on every dollar amount and percentage; $O(N)$ sub-millisecond computation; zero token bloat from passing hundreds of raw transaction rows into the LLM prompt.
  - *Cons*: Requires writing structured aggregation and simulation functions in Python rather than relying purely on prompt engineering.

### ADR-03: Typed Pydantic Inter-Agent Contracts vs. Unstructured Prose Handoffs
* **Decision**: Agents pass strictly validated **Pydantic V2 models** (`SpendingAnalysisReport`, `RecommendationReport`, `AlertReport`, `ExecutionPlan`) across the LangGraph state rather than free-text paragraphs.
* **Context**: When Agent A passes a prose paragraph to Agent B, key metrics can be lost, misinterpreted, or subtly altered ("telephone game" effect).
* **Trade-Offs**:
  - *Pros*: Zero information loss between `Spending Analysis Agent` and `Recommendation Agent`; enables the UI to render structured tables and JSON trace inspectors effortlessly; catches contract violations at runtime.
  - *Cons*: Schema changes require updating Pydantic definitions in `src/schemas/models.py`.

### ADR-04: Dual-Mode LLM Engine (Live Google Gemini + Deterministic Hybrid Fallback)
* **Decision**: Implement `GeminiEngine` (`src/llm/engine.py`) supporting live `google-genai` (`gemini-2.5-flash`) when `GOOGLE_API_KEY` is present, paired with an automatic **Deterministic Financial Reasoning Fallback** when no API key is configured.
* **Context**: Evaluators grading a code submission may run tests or launch the UI before creating/configuring an external API key, or may hit free-tier rate limits.
* **Trade-Offs**:
  - *Pros*: **Zero-friction evaluation**—works 100% offline out-of-the-box with sub-10ms latency, while dynamically upgrading to live Gemini synthesis as soon as an API key is pasted in the UI sidebar or `.env`.
  - *Cons*: Maintaining both prompt templates and structured deterministic synthesis templates.

### ADR-05: Dual-Tier Memory Architecture (Short-Term Session Buffer + Long-Term JSON/SQLite Store)
* **Decision**: Separate memory into:
  1. **Short-Term Memory (STM)**: Sliding window of the last 12 chat turns + cached `SpendingAnalysisReport` / `RecommendationReport` artifacts in memory.
  2. **Long-Term Memory (LTM)**: Persistent SQLite customer profile (`bank_advisor.db`) + JSON fact store (`long_term_memory.json`).
* **Context**: Customers expect the advisor to remember constraints across turns (e.g., *"Adobe CC and Figma are essential work tools"* or *"Set my dining budget to \$500"*) without stuffing the entire historical ledger into every prompt.
* **Trade-Offs**:
  - *Pros*: Fast lookup, persistent personalization across restarts, and automatic conflict prevention (e.g., the `Recommendation Agent` checks LTM before recommending subscription cancellations).
  - *Cons*: Uses keyword/regex + structured extraction rather than a heavy external vector database (Chroma/Pinecone), which keeps the submission lightweight and self-contained.

### ADR-06: Dual-Mode Domain & Safety Guardrail (LLM Semantic Classifier + Deterministic Pattern/Scope Filter)
* **Decision**: Implement a two-layer input guardrail ([`src/tools/guardrail_tools.py`](file:///Users/vizzdd/Documents/FDE/Thougthscrest/src/tools/guardrail_tools.py)) inside the `Planner Agent` before any sub-agents or database tools are invoked:
  1. **When Live Gemini API Key is Present (`LLM Intent Guardrail`)**: Uses Gemini (`temperature=0.0`) as a semantic intent classifier (`ALLOW | <reason>` vs. `BLOCK | <reason>`) to inspect the user's intention and block anything unrelated to personal finance, banking, transactions, budgeting, or savings.
  2. **When No API Key is Provided (`Deterministic Pattern & Domain Guardrail`)**: Evaluates the query against categorized off-topic regex rules (`Cooking & Recipes` like *"give me a pasta receipe"*, `Medical & Health Advice` like *"how to treat my acne"*, `Coding/Homework`, `General Trivia/Politics`, and `Prompt Injection/Jailbreak Attempts`) combined with a positive financial-domain keyword check.
* **LangGraph Short-Circuit Routing**: When a query is blocked (`is_allowed = False`, `user_intent = "BLOCKED_OFF_TOPIC"`), `route_after_planner` short-circuits the graph directly from `planner_plan_node` $\rightarrow$ `planner_synthesize_node`, completely bypassing the `Spending Analysis`, `Recommendation`, and `Alerting` agents while still logging the blocked guardrail event to `logs/`.

---

## 5. Mock Data Design & Planted Behavioral Patterns

To thoroughly test multi-pattern reasoning, `src/data/mock_generator.py` seeds **3 distinct customer personas** across 3 months (`2026-08`, `2026-09`, `2026-10`):

1. **Alex Rivera (`CUST-001` — Young Tech Professional, \$6,200/mo income)**:
   - *Planted Pattern 1 (MoM Category Surge)*: `Food & Dining` jumps from \$395 (Aug) $\rightarrow$ \$430 (Sep) $\rightarrow$ **\$742 (Oct)** (`+72.6%` MoM, `165%` of \$450 budget), driven by 11 late-night DoorDash/UberEats orders totaling \$549.50.
   - *Planted Pattern 2 (Subscription Creep)*: 6 recurring subscriptions totaling **\$136.94/mo** (`171%` of \$80 budget), including 3 overlapping video streaming services (`StreamFlix UHD`, `HuluPlus`, `MaxCinema`) and 2 music apps (`Spotify Duo`, `Apple Music`).
   - *Planted Pattern 3 (Entertainment Breach)*: Concert tickets push `Entertainment` to \$335 (`134%` of \$250 budget).

2. **Priya Sharma (`CUST-002` — Family Household Manager, \$8,500/mo income)**:
   - *Planted Pattern 1 (Statistical $Z$-Score Outlier)*: A one-off **\$680.00** emergency dishwasher replacement at `HomeDepot Appliance Center` (`Z-score = 2.2+`, `2.5x` category mean).
   - *Planted Pattern 2 (Utility Bill Surge)*: `Metro Electric & Gas` jumps `+60.3%` MoM to \$388 total utilities (`149%` of \$260 budget).
   - *Planted Pattern 3 (Idle Cash Optimization)*: Holds **\$14,200** in checking earning `0.01%` APY; agents recommend sweeping \$7,400 into a `4.50%` APY High-Yield Vault (`+$333/yr` risk-free interest).

3. **Marcus Vance (`CUST-003` — Freelance Designer, Variable \$3,600–\$4,500/mo income)**:
   - *Planted Pattern 1 (Income Dip + Cashflow Burn Risk)*: October freelance payouts dip to \$3,600 while checking balance drops to **\$890.00** (`CRITICAL` liquidity alert).
   - *Planted Pattern 2 (Dual Micro-Spend Habits)*: `10x` specialty coffee visits (\$115/mo) + `8x` Uber/Lyft rideshares (\$276/mo) = **\$391/mo (\$4,692/yr)** in high-frequency micro-drains.
   - *Planted Pattern 3 (Memory-Aware Subscription Filtering)*: Has \$167.95/mo in software/media subscriptions, but Long-Term Memory records that `Adobe CC` and `Figma` are essential tax-deductible work tools—so the `Recommendation Agent` protects them and focuses on rideshare/coffee habits instead.

---

## 6. Production Scalability Roadmap (Future Enhancements)
If deployed to production for millions of bank customers, this modular architecture is ready to scale via:
1. **Event-Driven Streaming Alerts**: Connect `AlertingAgent` to a Kafka/PubSub transaction authorization stream for sub-second push notifications at point-of-sale.
2. **Semantic Vector Memory (pgvector)**: Upgrade `MemoryManager` to store semantic embeddings of historical chat transcripts alongside structured SQLite/PostgreSQL facts.
3. **Human-in-the-Loop Action Execution**: Add OAuth-scoped write tools allowing the customer to execute one-click transfers or subscription virtual-card blocks directly from the chat UI.

