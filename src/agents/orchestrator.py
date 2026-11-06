"""
LangGraph Multi-Agent Orchestrator (`StateGraph`) for the Personal Finance Advisor.

Topology & Enforced Dependency Contract:
1. `planner_plan_node`: Parses query, loads Dual-Tier Memory, creates `ExecutionPlan`.
2. `spending_analysis_node`: Invokes `SpendingAnalysisAgent` -> populates `state["spending_report"]`.
3. Conditional Edge (`route_after_spending`):
   - If `execution_plan.run_recommendation_agent` is True -> routes to `recommendation_node`,
     passing `state["spending_report"]` directly as required by the system architecture.
   - Else -> routes to `alerting_node` or `planner_synthesize_node`.
4. `recommendation_node`: Invokes `RecommendationAgent` strictly using `state["spending_report"]`.
5. `alerting_node`: Invokes `AlertingAgent` for proactive/reactive overspending and anomaly checks.
6. `planner_synthesize_node`: Collates all agent outputs, updates Short-Term & Long-Term Memory,
   and produces the final conversational response.
"""

from datetime import datetime
import json
from typing import Any, Dict, Literal
import uuid
from langgraph.graph import END, START, StateGraph
from src.agents.alerting_agent import AlertingAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.spending_agent import SpendingAnalysisAgent
from src.config import TRACE_LOG_DIR
from src.memory.memory_manager import memory_manager
from src.schemas.models import (
    AlertReport,
    ExecutionPlan,
    FinanceAdvisorState,
    RecommendationReport,
    SpendingAnalysisReport,
)


planner_agent = PlannerAgent()
spending_agent = SpendingAnalysisAgent()
recommendation_agent = RecommendationAgent()
alerting_agent = AlertingAgent()


# --- LangGraph Node Functions ---

def planner_plan_node(state: FinanceAdvisorState) -> Dict[str, Any]:
    """Node 1: Planner Agent understands query, loads memory, and builds the ExecutionPlan."""
    customer_id = state["customer_id"]
    user_query = state["user_query"]

    memory_context = memory_manager.get_long_term_memory(customer_id)
    chat_history = memory_manager.get_chat_history(customer_id)

    plan, trace_step = planner_agent.create_plan(
        customer_id=customer_id,
        user_query=user_query,
        memory_context=memory_context,
    )

    traces = list(state.get("agent_trace", []))
    traces.append(trace_step.model_dump())

    return {
        "memory_context": memory_context,
        "chat_history": chat_history,
        "execution_plan": plan.model_dump(),
        "agent_trace": traces,
    }


def spending_analysis_node(state: FinanceAdvisorState) -> Dict[str, Any]:
    """Node 2: Spending Analysis Agent reads transactions, categorizes spend, and generates summary insights."""
    customer_id = state["customer_id"]
    user_query = state["user_query"]
    traces = list(state.get("agent_trace", []))

    report, trace_step = spending_agent.run(
        customer_id=customer_id,
        user_query=user_query,
        step_order=len(traces) + 1,
    )
    traces.append(trace_step.model_dump())

    return {
        "spending_report": report.model_dump(),
        "agent_trace": traces,
    }


def recommendation_node(state: FinanceAdvisorState) -> Dict[str, Any]:
    """
    Node 3: Recommendation Agent.
    STRICT DEPENDENCY: Consumes `state["spending_report"]` produced by `spending_analysis_node`.
    """
    spending_dict = state.get("spending_report")
    if not spending_dict:
        raise RuntimeError(
            "LangGraph State Error: recommendation_node invoked without `spending_report` from Spending Analysis Agent."
        )

    spending_report = SpendingAnalysisReport.model_validate(spending_dict)
    plan_dict = state.get("execution_plan") or {}
    target_override = plan_dict.get("target_savings_override")
    traces = list(state.get("agent_trace", []))

    rec_report, trace_step = recommendation_agent.run(
        spending_report=spending_report,
        user_query=state.get("user_query", ""),
        target_savings_override=target_override,
        step_order=len(traces) + 1,
    )
    traces.append(trace_step.model_dump())

    return {
        "recommendation_report": rec_report.model_dump(),
        "agent_trace": traces,
    }


def alerting_node(state: FinanceAdvisorState) -> Dict[str, Any]:
    """Node 4: Alerting Agent detects overspending behaviour, budget breaches, and anomalies."""
    customer_id = state["customer_id"]
    traces = list(state.get("agent_trace", []))

    alert_report, trace_step = alerting_agent.run(
        customer_id=customer_id,
        step_order=len(traces) + 1,
    )
    traces.append(trace_step.model_dump())

    return {
        "alert_report": alert_report.model_dump(),
        "agent_trace": traces,
    }


def planner_synthesize_node(state: FinanceAdvisorState) -> Dict[str, Any]:
    """Node 5: Planner Agent collates outputs from all agents and updates conversation memory."""
    customer_id = state["customer_id"]
    user_query = state["user_query"]
    memory_context = state.get("memory_context") or memory_manager.get_long_term_memory(customer_id)
    chat_history = state.get("chat_history") or []
    traces = list(state.get("agent_trace", []))

    plan = ExecutionPlan.model_validate(state["execution_plan"])
    spending_report = (
        SpendingAnalysisReport.model_validate(state["spending_report"])
        if state.get("spending_report")
        else None
    )
    rec_report = (
        RecommendationReport.model_validate(state["recommendation_report"])
        if state.get("recommendation_report")
        else None
    )
    alert_report = (
        AlertReport.model_validate(state["alert_report"])
        if state.get("alert_report")
        else None
    )

    final_text, trace_step = planner_agent.synthesize_response(
        customer_id=customer_id,
        user_query=user_query,
        memory_context=memory_context,
        plan=plan,
        spending_report=spending_report,
        recommendation_report=rec_report,
        alert_report=alert_report,
        chat_history=chat_history,
        step_order=len(traces) + 1,
    )
    traces.append(trace_step.model_dump())

    # Persist conversation turn & artifacts in Short-Term Memory
    memory_manager.add_chat_turn(customer_id, "user", user_query)
    memory_manager.add_chat_turn(customer_id, "assistant", final_text)
    memory_manager.cache_session_artifacts(
        customer_id=customer_id,
        spending_report=state.get("spending_report"),
        recommendation_report=state.get("recommendation_report"),
        alert_report=state.get("alert_report"),
    )

    return {
        "final_response": final_text,
        "agent_trace": traces,
    }


# --- Conditional Edge Routers ---

def route_after_planner(
    state: FinanceAdvisorState,
) -> Literal["spending_analysis_node", "alerting_node", "planner_synthesize_node"]:
    """
    Route after Planner creates the ExecutionPlan:
    - If Guardrail blocked the query (`is_allowed == False`), short-circuit directly to
      `planner_synthesize_node` without running Spending Analysis, Recommendation, or Alerting agents!
    - Whenever `run_recommendation_agent` is True, ALWAYS route to `spending_analysis_node` first
      so `spending_report` is generated for the Recommendation Agent.
    """
    plan = state.get("execution_plan") or {}
    if not plan.get("is_allowed", True) or plan.get("user_intent") == "BLOCKED_OFF_TOPIC":
        return "planner_synthesize_node"
    if plan.get("run_spending_agent", True) or plan.get("run_recommendation_agent", True):
        return "spending_analysis_node"
    return "alerting_node"


def route_after_spending(
    state: FinanceAdvisorState,
) -> Literal["recommendation_node", "alerting_node", "planner_synthesize_node"]:
    """Route from Spending Analysis Agent -> Recommendation Agent (when recommendations needed) or Alerting."""
    plan = state.get("execution_plan") or {}
    if plan.get("run_recommendation_agent", True):
        return "recommendation_node"
    if plan.get("run_alerting_agent", True):
        return "alerting_node"
    return "planner_synthesize_node"


def route_after_recommendation(
    state: FinanceAdvisorState,
) -> Literal["alerting_node", "planner_synthesize_node"]:
    """Route from Recommendation Agent -> Alerting Agent or directly to Planner Synthesis."""
    plan = state.get("execution_plan") or {}
    if plan.get("run_alerting_agent", True):
        return "alerting_node"
    return "planner_synthesize_node"


def build_finance_advisor_graph():
    """Compile and return the LangGraph StateGraph for the multi-agent finance advisor."""
    workflow = StateGraph(FinanceAdvisorState)

    # Register the 4 agent nodes (+ planner split into entry planning & final collation)
    workflow.add_node("planner_plan_node", planner_plan_node)
    workflow.add_node("spending_analysis_node", spending_analysis_node)
    workflow.add_node("recommendation_node", recommendation_node)
    workflow.add_node("alerting_node", alerting_node)
    workflow.add_node("planner_synthesize_node", planner_synthesize_node)

    # Wire edges
    workflow.add_edge(START, "planner_plan_node")
    workflow.add_conditional_edges("planner_plan_node", route_after_planner)
    workflow.add_conditional_edges("spending_analysis_node", route_after_spending)
    workflow.add_conditional_edges("recommendation_node", route_after_recommendation)
    workflow.add_edge("alerting_node", "planner_synthesize_node")
    workflow.add_edge("planner_synthesize_node", END)

    return workflow.compile()


# Compiled singleton graph
advisor_graph = build_finance_advisor_graph()


def _persist_trace_to_log_folder(result_state: FinanceAdvisorState) -> str:
    """
    Persist the complete LangGraph multi-agent execution trace and inter-agent payloads
    to the `logs/` folder (`logs/agent_traces.jsonl` and `logs/<customer_id>_traces.log`)
    so the customer UI stays clean while developers/evaluators have full audit logs.
    """
    TRACE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    trace_id = f"TRC-{uuid.uuid4().hex[:8].upper()}"
    cid = result_state.get("customer_id", "UNKNOWN")
    query = result_state.get("user_query", "")
    trace_steps = result_state.get("agent_trace", [])

    # 1. Structured JSONL audit log (machine-readable)
    jsonl_record = {
        "trace_id": trace_id,
        "timestamp": ts,
        "customer_id": cid,
        "user_query": query,
        "execution_plan": result_state.get("execution_plan"),
        "agent_trace": trace_steps,
        "spending_report_summary": {
            "total_spend": (result_state.get("spending_report") or {}).get("total_current_month_spend"),
            "savings_rate_pct": (result_state.get("spending_report") or {}).get("savings_rate_pct"),
            "overspent_categories": (result_state.get("spending_report") or {}).get("top_overspent_categories"),
        },
        "recommendation_summary": {
            "monthly_savings": (result_state.get("recommendation_report") or {}).get("total_potential_monthly_savings"),
            "annual_savings": (result_state.get("recommendation_report") or {}).get("total_potential_annual_savings"),
            "new_savings_rate_pct": (result_state.get("recommendation_report") or {}).get("projected_new_savings_rate_pct"),
        },
    }
    jsonl_path = TRACE_LOG_DIR / "agent_traces.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(jsonl_record) + "\n")

    # 2. Human-readable per-customer trace log file for debugging & submission review
    readable_path = TRACE_LOG_DIR / f"{cid}_traces.log"
    lines = [
        "=" * 92,
        f"TRACE ID: {trace_id} | TIMESTAMP: {ts} | CUSTOMER: {cid}",
        f"USER QUERY: {query}",
        "-" * 92,
    ]
    for step in trace_steps:
        tools_str = ", ".join(step.get("tools_invoked", [])) or "None"
        lines.append(
            f"  Step {step['step_order']}: {step['agent_name']} [{step['action_type']}] "
            f"({step['latency_ms']} ms | {step['llm_mode_used']})"
        )
        lines.append(f"    • Summary : {step['summary']}")
        lines.append(f"    • Tools   : {tools_str}")
        lines.append(f"    • Handoff : {step['input_summary']} -> {step['output_summary']}")
    lines.append("=" * 92 + "\n")

    with open(readable_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(readable_path)


def run_advisor_turn(customer_id: str, user_query: str) -> FinanceAdvisorState:
    """Execute a complete multi-agent conversational turn through the LangGraph pipeline and log trace to `logs/`."""
    initial_state: FinanceAdvisorState = {
        "customer_id": customer_id,
        "user_query": user_query,
        "agent_trace": [],
    }
    result: FinanceAdvisorState = advisor_graph.invoke(initial_state)
    _persist_trace_to_log_folder(result)
    return result


def get_proactive_session_alerts(customer_id: str) -> AlertReport:
    """
    Run the Alerting Agent proactively when a customer opens the banking chatbot,
    upgrading the legacy static notification experience into intelligent proactive alerts.
    """
    report, _ = alerting_agent.run(customer_id=customer_id, step_order=1)
    return report

