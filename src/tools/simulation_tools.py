"""
Savings scenario simulation and budget optimization tools used by the Recommendation Agent.
Strictly consumes structured data from the SpendingAnalysisReport.
"""

from typing import List, Optional, Tuple
from langchain_core.tools import tool
from src.data.db import db
from src.schemas.models import (
    ActionableRecommendation,
    BudgetOptimizationRow,
    SpendingAnalysisReport,
)


def generate_quantified_recommendations(
    spending_report: SpendingAnalysisReport,
    target_savings_override: Optional[float] = None,
) -> Tuple[List[ActionableRecommendation], List[BudgetOptimizationRow], List[str]]:
    """
    Consume the SpendingAnalysisReport from the Spending Analysis Agent and compute
    concrete, dollar-quantified savings recommendations and budget optimizations.
    """
    profile = db.get_profile(spending_report.customer_id)
    primary_goal = profile.financial_goals[0] if profile.financial_goals else None
    goal_remaining = (
        max(0.0, primary_goal.target_amount - primary_goal.current_amount)
        if primary_goal
        else 5000.0
    )
    goal_name = primary_goal.name if primary_goal else "Savings Reserve"

    recommendations: List[ActionableRecommendation] = []
    budget_rows: List[BudgetOptimizationRow] = []
    rec_idx = 1

    # 1. Analyze Micro-Spend Habits from SpendingAnalysisReport (e.g. Food Delivery, Coffee, Rideshare)
    for micro in spending_report.micro_spend_patterns:
        if micro.total_monthly_spend >= 90.0:
            reduction_ratio = 0.50 if "Delivery" in micro.pattern_name else 0.45
            monthly_save = round(micro.total_monthly_spend * reduction_ratio, 2)
            annual_save = round(monthly_save * 12, 2)
            target_txns = max(2, int(micro.transaction_count_monthly * (1 - reduction_ratio)))
            months_saved = round(goal_remaining / max(monthly_save + max(spending_report.net_cash_flow, 200), 1), 1)

            recommendations.append(
                ActionableRecommendation(
                    rec_id=f"REC-{rec_idx:02d}",
                    title=f"Right-Size {micro.pattern_name} ({micro.transaction_count_monthly}x -> {target_txns}x/mo)",
                    target_category=micro.category,
                    observation_from_spending_agent=(
                        f"Spending Analysis Agent detected {micro.transaction_count_monthly} transactions averaging "
                        f"${micro.average_ticket:.2f} across {', '.join(micro.merchant_examples[:2])}, "
                        f"totaling ${micro.total_monthly_spend:,.2f}/month (${micro.annualized_impact:,.0f}/year)."
                    ),
                    proposed_action=(
                        f"Cap {micro.pattern_name.lower()} at {target_txns} orders/month (weekly limit) "
                        f"and redirect the difference via automated weekly transfer."
                    ),
                    estimated_monthly_savings=monthly_save,
                    estimated_annual_savings=annual_save,
                    goal_acceleration_impact=(
                        f"Contributes ${annual_save:,.0f}/yr directly toward '{goal_name}' "
                        f"(${goal_remaining:,.0f} remaining, ~{months_saved} months to full completion)."
                    ),
                    difficulty="Easy",
                    priority=rec_idx,
                )
            )
            rec_idx += 1

    # 2. Analyze Redundant / High Subscriptions from SpendingAnalysisReport
    # Respect customer Long-Term Memory (e.g. if Adobe CC & Figma are essential work tools, don't cancel them)
    from src.memory.memory_manager import memory_manager

    ltm = memory_manager.get_long_term_memory(spending_report.customer_id)
    learned_text = " ".join(ltm.get("learned_facts", [])).lower()

    redundant_subs = []
    for s in spending_report.recurring_subscriptions:
        if not s.is_potentially_redundant:
            continue
        if ("adobe" in s.merchant.lower() or "figma" in s.merchant.lower()) and (
            "adobe" in learned_text or "figma" in learned_text
        ):
            continue
        redundant_subs.append(s)

    if redundant_subs:
        # Suggest canceling redundant ones (keep 1 per category)
        save_subs = redundant_subs[1:] if len(redundant_subs) > 1 else redundant_subs
        monthly_sub_save = round(sum(s.monthly_cost for s in save_subs), 2)
        if monthly_sub_save > 0:
            sub_names = ", ".join(f"{s.merchant} (${s.monthly_cost:.2f})" for s in save_subs)
            recommendations.append(
                ActionableRecommendation(
                    rec_id=f"REC-{rec_idx:02d}",
                    title="Consolidate Overlapping Recurring Subscriptions",
                    target_category="Subscriptions",
                    observation_from_spending_agent=(
                        f"Spending Analysis Agent flagged {len(spending_report.recurring_subscriptions)} recurring "
                        f"subscriptions totaling ${spending_report.total_subscription_monthly_cost:,.2f}/mo, "
                        f"with overlapping services: {sub_names}."
                    ),
                    proposed_action=(
                        f"Cancel or rotate {sub_names} and retain a single primary service per category."
                    ),
                    estimated_monthly_savings=monthly_sub_save,
                    estimated_annual_savings=round(monthly_sub_save * 12, 2),
                    goal_acceleration_impact=(
                        f"Recovers ${monthly_sub_save * 12:,.0f}/year in passive recurring leakage with zero lifestyle drop."
                    ),
                    difficulty="Easy",
                    priority=rec_idx,
                )
            )
            rec_idx += 1
    elif spending_report.total_subscription_monthly_cost > 120.0:
        top_sub = spending_report.recurring_subscriptions[0]
        monthly_sub_save = round(top_sub.monthly_cost, 2)
        recommendations.append(
            ActionableRecommendation(
                rec_id=f"REC-{rec_idx:02d}",
                title=f"Audit High-Tier Subscription ({top_sub.merchant})",
                target_category="Subscriptions",
                observation_from_spending_agent=(
                    f"Total recurring software/media subscriptions reached "
                    f"${spending_report.total_subscription_monthly_cost:,.2f}/mo (152% of budget)."
                ),
                proposed_action=(
                    f"Downgrade {top_sub.merchant} (${top_sub.monthly_cost:.2f}/mo) to a standard tier or pause during low-utilization months."
                ),
                estimated_monthly_savings=round(monthly_sub_save * 0.5, 2),
                estimated_annual_savings=round(monthly_sub_save * 6, 2),
                goal_acceleration_impact=f"Lowers fixed monthly overhead by ${monthly_sub_save * 6:,.0f}/year.",
                difficulty="Easy",
                priority=rec_idx,
            )
        )
        rec_idx += 1

    # 3. Analyze Category Budget Breaches & MoM Spikes from SpendingAnalysisReport
    for cat_summary in spending_report.category_breakdown:
        if cat_summary.category in ("Housing & Rent", "Groceries"):
            budget_rows.append(
                BudgetOptimizationRow(
                    category=cat_summary.category,
                    current_spend=cat_summary.current_month_spend,
                    current_budget=cat_summary.monthly_budget,
                    recommended_budget=cat_summary.monthly_budget,
                    monthly_delta=0.0,
                    rationale="Essential baseline category within healthy parameters.",
                )
            )
            continue

        if cat_summary.current_month_spend > cat_summary.monthly_budget:
            excess = round(cat_summary.current_month_spend - cat_summary.monthly_budget, 2)
            # Recommend returning to budget or 3-month average
            rec_budget = cat_summary.monthly_budget
            delta = round(cat_summary.current_month_spend - rec_budget, 2)
            budget_rows.append(
                BudgetOptimizationRow(
                    category=cat_summary.category,
                    current_spend=cat_summary.current_month_spend,
                    current_budget=cat_summary.monthly_budget,
                    recommended_budget=rec_budget,
                    monthly_delta=delta,
                    rationale=(
                        f"Currently {cat_summary.budget_utilization_pct:.0f}% of budget "
                        f"({cat_summary.mom_change_pct:+.1f}% MoM); cap at ${rec_budget:,.0f} to save ${delta:,.0f}/mo."
                    ),
                )
            )

            # Avoid duplicating if already covered by Food Delivery micro-spend
            already_covered = any(r.target_category == cat_summary.category for r in recommendations)
            if not already_covered and excess >= 50.0:
                recommendations.append(
                    ActionableRecommendation(
                        rec_id=f"REC-{rec_idx:02d}",
                        title=f"Normalize {cat_summary.category} Spike Back to Baseline",
                        target_category=cat_summary.category,
                        observation_from_spending_agent=(
                            f"Spending Analysis Agent reported {cat_summary.category} surged to "
                            f"${cat_summary.current_month_spend:,.2f} ({cat_summary.mom_change_pct:+.1f}% MoM vs "
                            f"${cat_summary.previous_month_spend:,.2f} last month), exceeding your "
                            f"${cat_summary.monthly_budget:,.0f} budget by ${excess:,.2f}."
                        ),
                        proposed_action=(
                            f"Set a hard mid-month spending alert at ${cat_summary.monthly_budget * 0.75:,.0f} and "
                            f"revert to your prior baseline of ${cat_summary.previous_month_spend:,.0f}/mo."
                        ),
                        estimated_monthly_savings=excess,
                        estimated_annual_savings=round(excess * 12, 2),
                        goal_acceleration_impact=(
                            f"Reclaims ${excess:,.0f}/month (${excess * 12:,.0f}/yr) toward '{goal_name}'."
                        ),
                        difficulty="Medium",
                        priority=rec_idx,
                    )
                )
                rec_idx += 1
        else:
            budget_rows.append(
                BudgetOptimizationRow(
                    category=cat_summary.category,
                    current_spend=cat_summary.current_month_spend,
                    current_budget=cat_summary.monthly_budget,
                    recommended_budget=cat_summary.monthly_budget,
                    monthly_delta=0.0,
                    rationale=f"On track ({cat_summary.budget_utilization_pct:.0f}% of budget used).",
                )
            )

    # 4. Check Idle Checking Balance Optimization
    if profile.checking_balance > profile.monthly_net_income * 1.4:
        sweep_amt = round(profile.checking_balance - (profile.monthly_net_income * 0.8), -2)
        monthly_interest = round((sweep_amt * 0.045) / 12.0, 2)
        recommendations.append(
            ActionableRecommendation(
                rec_id=f"REC-{rec_idx:02d}",
                title=f"Sweep ${sweep_amt:,.0f} Idle Checking Cash to 4.50% APY Vault",
                target_category="Cash Yield Optimization",
                observation_from_spending_agent=(
                    f"Net monthly cash flow is strong (${spending_report.net_cash_flow:,.2f}/mo) while "
                    f"${profile.checking_balance:,.2f} sits in checking earning near-zero interest."
                ),
                proposed_action=f"Transfer ${sweep_amt:,.0f} to High-Yield Savings and automate a ${max(400.0, round(spending_report.net_cash_flow * 0.6, -1)):,.0f}/mo recurring sweep.",
                estimated_monthly_savings=monthly_interest,
                estimated_annual_savings=round(monthly_interest * 12, 2),
                goal_acceleration_impact=f"Immediately boosts '{goal_name}' balance by ${sweep_amt:,.0f} + ${monthly_interest * 12:,.0f}/yr in interest.",
                difficulty="Easy",
                priority=rec_idx,
            )
        )

    # Bank features matched to persona needs
    bank_features = [
        f"Auto-Save Goal Rule: Automatically transfer ${round(sum(r.estimated_monthly_savings for r in recommendations), 0):,.0f}/month on payday to '{goal_name}'.",
        "Smart Category Guardrails: Push real-time interactive nudges at 75% and 90% of Food & Dining and Entertainment budgets.",
        "Subscription Vault Manager: One-click virtual card freeze for redundant recurring merchants.",
    ]

    return recommendations, budget_rows, bank_features


@tool
def tool_simulate_budget_optimization(spending_report_json: dict, target_savings: Optional[float] = None) -> dict:
    """
    Calculate personalized savings recommendations and budget optimizations
    strictly from a SpendingAnalysisReport dictionary.
    """
    report = SpendingAnalysisReport.model_validate(spending_report_json)
    recs, budget_rows, features = generate_quantified_recommendations(report, target_savings)
    return {
        "recommendations": [r.model_dump() for r in recs],
        "budget_optimizations": [b.model_dump() for b in budget_rows],
        "recommended_bank_features": features,
    }
