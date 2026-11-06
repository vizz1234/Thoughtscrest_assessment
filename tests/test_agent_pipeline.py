"""Unit and integration tests for the Multi-Agent Personal Finance Advisor."""

import unittest
from src.agents.orchestrator import get_proactive_session_alerts, run_advisor_turn
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.spending_agent import SpendingAnalysisAgent
from src.data.db import db
from src.memory.memory_manager import memory_manager


class TestAgenticFinanceAdvisor(unittest.TestCase):
    """End-to-end test suite verifying mock data, tools, LangGraph state handoffs, and memory."""

    def setUp(self):
        db.seed_mock_data()
        memory_manager.clear_session("CUST-001")
        memory_manager.clear_session("CUST-002")
        memory_manager.clear_session("CUST-003")

    def test_1_mock_profiles_and_transactions_seeded(self):
        profiles = db.list_profiles()
        self.assertEqual(len(profiles), 3)
        alex_txns = db.query_transactions("CUST-001", transaction_type="debit")
        self.assertGreater(len(alex_txns), 40)

    def test_2_spending_analysis_agent_detects_patterns(self):
        agent = SpendingAnalysisAgent()
        report, trace = agent.run("CUST-001", "Analyze my spending")
        self.assertEqual(report.customer_id, "CUST-001")
        self.assertIn("Food & Dining", report.top_overspent_categories)
        self.assertIn("Subscriptions", report.top_overspent_categories)
        self.assertGreater(report.total_subscription_monthly_cost, 130.0)
        self.assertGreaterEqual(len(report.micro_spend_patterns), 1)
        self.assertIn("tool_aggregate_spending_by_category", trace.tools_invoked)

    def test_3_recommendation_agent_requires_spending_report_dependency(self):
        rec_agent = RecommendationAgent()
        # Verify strict contract enforcement when SpendingAnalysisReport is missing
        with self.assertRaises(ValueError):
            rec_agent.run(spending_report=None)  # type: ignore

        # Verify proper handoff when SpendingAnalysisReport is provided
        spending_report, _ = SpendingAnalysisAgent().run("CUST-001")
        rec_report, trace = rec_agent.run(spending_report=spending_report)
        self.assertGreater(rec_report.total_potential_monthly_savings, 200.0)
        self.assertGreater(rec_report.projected_new_savings_rate_pct, spending_report.savings_rate_pct)
        self.assertEqual(trace.action_type, "HANDOFF")

    def test_4_proactive_alerting_agent_flags_breaches_and_anomalies(self):
        # Alex has budget breaches & subscription creep
        alex_alerts = get_proactive_session_alerts("CUST-001")
        self.assertEqual(alex_alerts.overall_risk_level, "HIGH")
        alert_types = {a.alert_type for a in alex_alerts.alerts}
        self.assertIn("BUDGET_BREACH", alert_types)
        self.assertIn("SUBSCRIPTION_CREEP", alert_types)

        # Priya has a $680 HomeDepot appliance Z-score spike & idle checking cash
        priya_alerts = get_proactive_session_alerts("CUST-002")
        priya_types = {a.alert_type for a in priya_alerts.alerts}
        self.assertIn("SPENDING_SPIKE", priya_types)

    def test_5_langgraph_full_orchestration_and_memory_persistence(self):
        query = "I'm saving for a wedding in December—set my dining budget to $500 and show me how to save more."
        state = run_advisor_turn("CUST-001", query)

        # Verify all 4 agents participated in the LangGraph execution trace
        agent_names = [step["agent_name"] for step in state["agent_trace"]]
        self.assertEqual(agent_names[0], "Planner Agent")
        self.assertIn("Spending Analysis Agent", agent_names)
        self.assertIn("Recommendation Agent", agent_names)
        self.assertIn("Alerting Agent", agent_names)
        self.assertEqual(agent_names[-1], "Planner Agent")

        # Verify Spending Analysis ran BEFORE Recommendation Agent in trace
        spend_idx = agent_names.index("Spending Analysis Agent")
        rec_idx = agent_names.index("Recommendation Agent")
        self.assertLess(spend_idx, rec_idx)

        # Verify Long-Term Memory persisted the budget change and wedding fact
        updated_profile = db.get_profile("CUST-001")
        self.assertEqual(updated_profile.monthly_budgets["Food & Dining"], 500.0)
        ltm = memory_manager.get_long_term_memory("CUST-001")
        self.assertTrue(any("500" in f for f in ltm["learned_facts"]))

    def test_6_customer_session_and_memory_isolation(self):
        # Run a query with a custom fact for CUST-001
        run_advisor_turn("CUST-001", "I am saving for a trip to Tokyo in December.")

        # Verify CUST-001 has chat history and the Tokyo fact
        cust1_history = memory_manager.get_chat_history("CUST-001")
        cust1_ltm = memory_manager.get_long_term_memory("CUST-001")
        self.assertEqual(len(cust1_history), 2)
        self.assertTrue(any("Tokyo" in f for f in cust1_ltm["learned_facts"]))

        # Verify CUST-002 has completely empty chat history and no Tokyo fact leakage
        cust2_history = memory_manager.get_chat_history("CUST-002")
        cust2_ltm = memory_manager.get_long_term_memory("CUST-002")
        self.assertEqual(len(cust2_history), 0)
        self.assertFalse(any("Tokyo" in f for f in cust2_ltm["learned_facts"]))

    def test_7_domain_guardrail_blocks_off_topic_queries(self):
        # 1. Test "give me a pasta receipe" -> must be blocked and skip sub-agents
        pasta_state = run_advisor_turn("CUST-001", "give me a pasta receipe")
        self.assertEqual(pasta_state["execution_plan"]["user_intent"], "BLOCKED_OFF_TOPIC")
        self.assertFalse(pasta_state["execution_plan"]["is_allowed"])
        self.assertIsNone(pasta_state.get("spending_report"))
        self.assertIsNone(pasta_state.get("recommendation_report"))
        self.assertIn("Outside Financial Advisory Scope", pasta_state["final_response"])

        # 2. Test "how to treat my acne" -> must be blocked and skip sub-agents
        acne_state = run_advisor_turn("CUST-001", "how to treat my acne")
        self.assertEqual(acne_state["execution_plan"]["user_intent"], "BLOCKED_OFF_TOPIC")
        self.assertFalse(acne_state["execution_plan"]["is_allowed"])
        self.assertIsNone(acne_state.get("spending_report"))
        self.assertIn("Outside Financial Advisory Scope", acne_state["final_response"])


if __name__ == "__main__":
    unittest.main()

