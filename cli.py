"""
Interactive CLI & Automated Demo Runner for the Personal Finance Advisor (`python cli.py`).
Supports:
- `python cli.py --demo`: Runs a full multi-persona showcase demonstrating all 4 agents,
  tool calling, LangGraph state handoffs (Spending Analysis -> Recommendation Agent),
  proactive alerting, and memory updates.
- `python cli.py`: Interactive conversational shell.
"""

import argparse
import sys
from src.agents.orchestrator import get_proactive_session_alerts, run_advisor_turn
from src.data.db import db
from src.memory.memory_manager import memory_manager


def print_banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(f"  {title}")
    print("=" * 88)


def print_trace(agent_trace: list) -> None:
    print("\n  [LangGraph Multi-Agent Execution Trace]")
    for step in agent_trace:
        tools_str = ", ".join(step.get("tools_invoked", [])) or "None"
        print(
            f"   ├─ Step {step['step_order']}: {step['agent_name']} [{step['action_type']}] "
            f"({step['latency_ms']} ms | {step['llm_mode_used']})"
        )
        print(f"   │    Summary : {step['summary']}")
        print(f"   │    Tools   : {tools_str}")
        print(f"   │    Handoff : {step['input_summary']} -> {step['output_summary']}")
    print("   └─ Pipeline Complete.\n")


def run_demo_walkthrough() -> None:
    """Execute 3 end-to-end evaluation scenarios across personas for submission verification."""
    db.seed_mock_data()

    print_banner("CRESTBANK AI PERSONAL FINANCE ADVISOR — MULTI-AGENT DEMO WALKTHROUGH")

    scenarios = [
        (
            "CUST-001",
            "Scenario 1 (Alex Rivera — Young Tech Professional): Full Spending Analysis -> Personalized Savings Plan",
            "Analyze my spending patterns this month, especially Food & Dining and Subscriptions, and tell me how I can save more toward my House Down Payment.",
        ),
        (
            "CUST-002",
            "Scenario 2 (Priya Sharma — Family Household Manager): Anomaly Detection, Utility Surge & Idle Cash Optimization",
            "Are there any unusual spending spikes or budget alerts on my account this month, and how can I optimize my family's cash and budget?",
        ),
        (
            "CUST-003",
            "Scenario 3 (Marcus Vance — Freelance Designer): Memory Update + Micro-Spend & Cashflow Protection",
            "I'm saving for a new studio laptop in May—set my food budget to $260 and show me how to stop my daily coffee and rideshare drain.",
        ),
    ]

    for cid, title, query in scenarios:
        profile = db.get_profile(cid)
        print_banner(f"{title}\n  Customer: {profile.name} ({cid}) | Income: ${profile.monthly_net_income:,.2f}/mo")

        # 1. Show Proactive Alerting Agent startup scan
        proactive = get_proactive_session_alerts(cid)
        print(f"  [Proactive Alerting Agent Startup Scan — Overall Risk: {proactive.overall_risk_level}]")
        for a in proactive.alerts:
            print(f"   • [{a.severity}] {a.title}: {a.metric_highlight} -> {a.recommended_immediate_action}")

        # 2. Run LangGraph conversational turn
        print(f"\n  USER QUERY: \"{query}\"")
        state = run_advisor_turn(cid, query)

        # 3. Print LangGraph trace
        print_trace(state.get("agent_trace", []))

        # 4. Print Final Planner Response
        print("  [FINAL PLANNER AGENT RESPONSE TO CUSTOMER]")
        print("-" * 88)
        print(state["final_response"])
        print("-" * 88)


def run_interactive_cli(customer_id: str = "CUST-001") -> None:
    """Run an interactive conversational loop in the terminal."""
    profile = db.get_profile(customer_id)
    print_banner(f"CrestBank Interactive Advisor — Active Persona: {profile.name} ({customer_id})")
    proactive = get_proactive_session_alerts(customer_id)
    print(f"Proactive Alert Banner ({proactive.overall_risk_level} Risk): {proactive.proactive_headline}\n")
    print("Type your question, '/switch CUST-002' to change persona, or 'exit' to quit.\n")

    while True:
        try:
            user_in = input(f"[{profile.name}] You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting Advisor CLI.")
            break

        if not user_in:
            continue
        if user_in.lower() in ("exit", "quit", "q"):
            break
        if user_in.startswith("/switch"):
            parts = user_in.split()
            if len(parts) > 1:
                customer_id = parts[1].upper()
                profile = db.get_profile(customer_id)
                print(f"Switched to {profile.name} ({customer_id}).")
            continue

        state = run_advisor_turn(customer_id, user_in)
        print_trace(state.get("agent_trace", []))
        print(f"\nAdvisor >\n{state['final_response']}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Personal Finance Advisor CLI")
    parser.add_argument("--demo", action="store_true", help="Run automated 3-scenario demo walkthrough")
    parser.add_argument("--customer", default="CUST-001", help="Customer ID (CUST-001, CUST-002, CUST-003)")
    args = parser.parse_args()

    if args.demo:
        run_demo_walkthrough()
    else:
        run_interactive_cli(args.customer)

