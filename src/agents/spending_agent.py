"""
Agent 2: Spending Analysis Agent
Reads transaction data, categorizes spending across Food & Dining, Groceries, Entertainment,
Subscriptions, Shopping, Utilities, etc., computes Month-over-Month (MoM) shifts, detects
recurring subscriptions and micro-spend habits, and outputs a structured `SpendingAnalysisReport`.
"""

import time
from typing import Tuple
from src.config import CURRENT_MONTH, PREVIOUS_MONTH
from src.llm.engine import llm_engine
from src.schemas.models import AgentTraceStep, SpendingAnalysisReport
from src.tools.transaction_tools import (
    compute_category_breakdown,
    detect_subscriptions_and_micro_spends,
)


class SpendingAnalysisAgent:
    """Specialized agent for transaction categorization, aggregation, and pattern recognition."""

    name = "Spending Analysis Agent"

    def run(
        self,
        customer_id: str,
        user_query: str = "",
        step_order: int = 2,
    ) -> Tuple[SpendingAnalysisReport, AgentTraceStep]:
        start_ts = time.perf_counter()

        # 1. Invoke deterministic transaction aggregation tool
        summaries, total_curr_spend, total_prev_spend, monthly_income = compute_category_breakdown(
            customer_id=customer_id,
            current_month=CURRENT_MONTH,
            previous_month=PREVIOUS_MONTH,
        )

        # 2. Invoke subscription and micro-spend pattern detection tool
        subscriptions, micro_patterns = detect_subscriptions_and_micro_spends(
            customer_id=customer_id,
            current_month=CURRENT_MONTH,
        )

        net_cash_flow = round(monthly_income - total_curr_spend, 2)
        savings_rate_pct = round((net_cash_flow / monthly_income) * 100.0, 1) if monthly_income > 0 else 0.0
        mom_total_change_pct = (
            round(((total_curr_spend - total_prev_spend) / total_prev_spend) * 100.0, 1)
            if total_prev_spend > 0
            else 0.0
        )

        top_overspent = [
            s.category
            for s in summaries
            if s.current_month_spend > s.monthly_budget and s.category != "Housing & Rent"
        ]
        fastest_growing = [
            f"{s.category} ({s.mom_change_pct:+.1f}% MoM)"
            for s in sorted(summaries, key=lambda x: x.mom_change_pct, reverse=True)
            if s.mom_change_pct >= 15.0 and s.category != "Housing & Rent"
        ]
        total_sub_cost = round(sum(s.monthly_cost for s in subscriptions), 2)

        # 3. Synthesize structured insights
        key_insights = [
            f"Total spending in {CURRENT_MONTH} was ${total_curr_spend:,.2f} against ${monthly_income:,.2f} net income "
            f"(Net Cash Flow: ${net_cash_flow:,.2f}, Savings Rate: {savings_rate_pct:.1f}%, {mom_total_change_pct:+.1f}% MoM).",
        ]

        for s in summaries[:4]:
            status = (
                f"OVER BUDGET by ${s.current_month_spend - s.monthly_budget:,.2f}"
                if s.current_month_spend > s.monthly_budget
                else f"Within budget ({s.budget_utilization_pct:.0f}% used)"
            )
            key_insights.append(
                f"{s.category}: ${s.current_month_spend:,.2f} ({s.share_of_total_spend_pct:.1f}% of spend, "
                f"{s.mom_change_pct:+.1f}% vs {PREVIOUS_MONTH}) — {status}."
            )

        if subscriptions:
            redundant_cnt = sum(1 for sub in subscriptions if sub.is_potentially_redundant)
            key_insights.append(
                f"Recurring Subscriptions: {len(subscriptions)} active charges totaling ${total_sub_cost:,.2f}/mo "
                f"(${total_sub_cost * 12:,.2f}/yr), with {redundant_cnt} overlapping/redundant services flagged."
            )

        for mp in micro_patterns:
            key_insights.append(
                f"Micro-Spend Pattern ({mp.pattern_name}): {mp.transaction_count_monthly} transactions averaging "
                f"${mp.average_ticket:.2f} = ${mp.total_monthly_spend:,.2f}/mo (${mp.annualized_impact:,.0f}/yr)."
            )

        # Optional LLM executive summary enrichment
        default_exec_summary = " ".join(key_insights[:3])
        llm_summary, mode_used = llm_engine.generate_text(
            system_prompt=(
                "You are the Spending Analysis Agent for a bank. Summarize the structured transaction metrics "
                "in 2 concise, analytical sentences citing exact dollar figures and percentages."
            ),
            user_prompt=f"Customer query: {user_query}\nStructured insights:\n" + "\n".join(key_insights),
        )
        exec_summary = llm_summary or default_exec_summary

        report = SpendingAnalysisReport(
            customer_id=customer_id,
            analysis_period=f"2026-08 to {CURRENT_MONTH}",
            current_month=CURRENT_MONTH,
            previous_month=PREVIOUS_MONTH,
            monthly_income=monthly_income,
            total_current_month_spend=total_curr_spend,
            total_previous_month_spend=total_prev_spend,
            net_cash_flow=net_cash_flow,
            savings_rate_pct=savings_rate_pct,
            mom_total_change_pct=mom_total_change_pct,
            category_breakdown=summaries,
            top_overspent_categories=top_overspent,
            fastest_growing_categories=fastest_growing,
            recurring_subscriptions=subscriptions,
            total_subscription_monthly_cost=total_sub_cost,
            micro_spend_patterns=micro_patterns,
            key_insights=key_insights,
            executive_summary=exec_summary,
        )

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
        trace = AgentTraceStep(
            step_order=step_order,
            agent_name=self.name,
            action_type="TOOL_CALL",
            summary=(
                f"Analyzed 3-month transaction ledger for {customer_id}: categorized ${total_curr_spend:,.2f} spend "
                f"across {len(summaries)} categories, flagged {len(top_overspent)} overspent categories "
                f"({', '.join(top_overspent) or 'None'}) and {len(subscriptions)} recurring subscriptions."
            ),
            tools_invoked=[
                "tool_aggregate_spending_by_category",
                "tool_detect_subscriptions_and_habits",
            ],
            input_summary=f"customer_id={customer_id}, period=2026-08..{CURRENT_MONTH}",
            output_summary=(
                f"Produced SpendingAnalysisReport (Total Spend=${total_curr_spend:,.2f}, "
                f"Savings Rate={savings_rate_pct:.1f}%, MoM={mom_total_change_pct:+.1f}%)"
            ),
            latency_ms=elapsed_ms,
            llm_mode_used=mode_used,
        )

        return report, trace

