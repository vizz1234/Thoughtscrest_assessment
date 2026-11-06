"""
Agent 3: Recommendation Agent
Strictly consumes the `SpendingAnalysisReport` produced by the Spending Analysis Agent
along with the customer's long-term profile & goals to generate quantified savings plans
and budget optimizations.
"""

import time
from typing import Optional, Tuple
from src.llm.engine import llm_engine
from src.schemas.models import (
    AgentTraceStep,
    RecommendationReport,
    SpendingAnalysisReport,
)
from src.tools.simulation_tools import generate_quantified_recommendations


class RecommendationAgent:
    """
    Specialized agent for personalized financial recommendations and budget optimization.
    ENFORCED ARCHITECTURAL CONTRACT: Must receive `SpendingAnalysisReport` from Spending Analysis Agent.
    """

    name = "Recommendation Agent"

    def run(
        self,
        spending_report: SpendingAnalysisReport,
        user_query: str = "",
        target_savings_override: Optional[float] = None,
        step_order: int = 3,
    ) -> Tuple[RecommendationReport, AgentTraceStep]:
        if spending_report is None:
            raise ValueError(
                "Architectural Contract Violation: RecommendationAgent requires a valid "
                "SpendingAnalysisReport from the SpendingAnalysisAgent as input."
            )

        start_ts = time.perf_counter()

        # Run deterministic budget & savings simulation tool on the SpendingAnalysisReport
        recommendations, budget_rows, bank_features = generate_quantified_recommendations(
            spending_report=spending_report,
            target_savings_override=target_savings_override,
        )

        total_monthly_savings = round(sum(r.estimated_monthly_savings for r in recommendations), 2)
        total_annual_savings = round(total_monthly_savings * 12, 2)

        new_net_cash_flow = spending_report.net_cash_flow + total_monthly_savings
        projected_new_savings_rate = (
            round((new_net_cash_flow / spending_report.monthly_income) * 100.0, 1)
            if spending_report.monthly_income > 0
            else 0.0
        )

        default_strategy = (
            f"Based on the Spending Analysis Agent's report (${spending_report.total_current_month_spend:,.2f} spend, "
            f"{spending_report.savings_rate_pct:.1f}% current savings rate), implementing these {len(recommendations)} "
            f"targeted adjustments unlocks **${total_monthly_savings:,.2f}/month (${total_annual_savings:,.2f}/year)** "
            f"and boosts your monthly savings rate from **{spending_report.savings_rate_pct:.1f}% to {projected_new_savings_rate:.1f}%**."
        )

        rec_bullets = "\n".join(
            f"- {r.title}: Save ${r.estimated_monthly_savings:.2f}/mo (${r.estimated_annual_savings:.0f}/yr) via {r.proposed_action}"
            for r in recommendations
        )
        llm_strategy, mode_used = llm_engine.generate_text(
            system_prompt=(
                "You are the Recommendation Agent for a bank. Write a 2-sentence strategic summary explaining "
                "how the proposed savings plan builds directly on the Spending Analysis Agent's findings."
            ),
            user_prompt=(
                f"User Query: {user_query}\n"
                f"Spending Report Summary: {spending_report.executive_summary}\n"
                f"Quantified Recommendations:\n{rec_bullets}"
            ),
        )

        report = RecommendationReport(
            customer_id=spending_report.customer_id,
            source_analysis_period=spending_report.analysis_period,
            total_potential_monthly_savings=total_monthly_savings,
            total_potential_annual_savings=total_annual_savings,
            projected_new_savings_rate_pct=projected_new_savings_rate,
            recommendations=recommendations,
            budget_optimizations=budget_rows,
            recommended_bank_features=bank_features,
            strategic_summary=llm_strategy or default_strategy,
        )

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
        trace = AgentTraceStep(
            step_order=step_order,
            agent_name=self.name,
            action_type="HANDOFF",
            summary=(
                f"Consumed SpendingAnalysisReport from Spending Analysis Agent -> generated {len(recommendations)} "
                f"personalized recommendations unlocking ${total_monthly_savings:,.2f}/mo "
                f"(${total_annual_savings:,.2f}/yr) and lifting savings rate to {projected_new_savings_rate:.1f}%."
            ),
            tools_invoked=["tool_simulate_budget_optimization"],
            input_summary=(
                f"SpendingAnalysisReport(total_spend=${spending_report.total_current_month_spend:,.2f}, "
                f"overspent={spending_report.top_overspent_categories}, "
                f"micro_patterns={len(spending_report.micro_spend_patterns)})"
            ),
            output_summary=(
                f"Produced RecommendationReport (Monthly Savings=${total_monthly_savings:,.2f}, "
                f"Annual=${total_annual_savings:,.2f}, New Savings Rate={projected_new_savings_rate:.1f}%)"
            ),
            latency_ms=elapsed_ms,
            llm_mode_used=mode_used,
        )

        return report, trace

