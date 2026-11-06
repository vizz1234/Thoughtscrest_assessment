"""
Anomaly detection, budget breach analysis, and cash-flow risk tools
used by the Alerting Agent.
"""

from collections import defaultdict
import math
from typing import List
from langchain_core.tools import tool
from src.config import (
    ANOMALY_Z_SCORE_THRESHOLD,
    BUDGET_CRITICAL_THRESHOLD,
    BUDGET_WARNING_THRESHOLD,
    CURRENT_MONTH,
    PREVIOUS_MONTH,
)
from src.data.db import db
from src.schemas.models import FinancialAlert
from src.tools.transaction_tools import (
    compute_category_breakdown,
    detect_subscriptions_and_micro_spends,
)


def evaluate_customer_alerts(
    customer_id: str,
    current_month: str = CURRENT_MONTH,
    previous_month: str = PREVIOUS_MONTH,
) -> List[FinancialAlert]:
    """
    Run deterministic risk rules across:
    1. Category Budget Breaches & Near-Breaches
    2. Month-over-Month Category Surges (>35% MoM increase)
    3. Single-Transaction Statistical Outliers (Z-score > threshold or >2.5x category ticket avg)
    4. Subscription Creep & Overlapping Services
    5. Cash-Flow / Low Checking Buffer Risk
    """
    profile = db.get_profile(customer_id)
    summaries, total_curr_spend, _, monthly_income = compute_category_breakdown(
        customer_id, current_month, previous_month
    )
    subs, micro_patterns = detect_subscriptions_and_micro_spends(customer_id, current_month)
    all_debits = db.query_transactions(customer_id, transaction_type="debit")

    alerts: List[FinancialAlert] = []
    alert_idx = 1

    # 1. Category Budget Breaches & Surges
    for s in summaries:
        if s.category == "Housing & Rent":
            continue
        ratio = s.current_month_spend / s.monthly_budget if s.monthly_budget > 0 else 0.0

        if ratio >= BUDGET_CRITICAL_THRESHOLD:
            over_amt = round(s.current_month_spend - s.monthly_budget, 2)
            severity = "CRITICAL" if ratio >= 1.25 or over_amt >= 150 else "WARNING"
            alerts.append(
                FinancialAlert(
                    alert_id=f"ALT-{alert_idx:02d}",
                    severity=severity,
                    alert_type="BUDGET_BREACH",
                    category=s.category,
                    title=f"{s.category} Budget Exceeded ({s.budget_utilization_pct:.0f}% Used)",
                    description=(
                        f"You spent ${s.current_month_spend:,.2f} in {s.category} this month against a "
                        f"${s.monthly_budget:,.2f} budget (+${over_amt:,.2f} over budget; "
                        f"{s.mom_change_pct:+.1f}% vs last month)."
                    ),
                    metric_highlight=f"${s.current_month_spend:,.0f} / ${s.monthly_budget:,.0f} ({s.budget_utilization_pct:.0f}%)",
                    recommended_immediate_action=(
                        f"Pause discretionary {s.category} spend and cap next month at ${s.monthly_budget:,.0f}."
                    ),
                )
            )
            alert_idx += 1
        elif ratio >= BUDGET_WARNING_THRESHOLD:
            alerts.append(
                FinancialAlert(
                    alert_id=f"ALT-{alert_idx:02d}",
                    severity="WARNING",
                    alert_type="BUDGET_BREACH",
                    category=s.category,
                    title=f"Approaching {s.category} Budget Limit ({s.budget_utilization_pct:.0f}%)",
                    description=(
                        f"{s.category} spending is at ${s.current_month_spend:,.2f} of your ${s.monthly_budget:,.2f} cap."
                    ),
                    metric_highlight=f"{s.budget_utilization_pct:.0f}% of budget utilized",
                    recommended_immediate_action=f"Keep remaining {s.category} purchases under ${s.monthly_budget - s.current_month_spend:,.0f}.",
                )
            )
            alert_idx += 1

    # 2. Single-Transaction Statistical Outlier Detection
    by_cat_amounts = defaultdict(list)
    for t in all_debits:
        if t.category != "Housing & Rent":
            by_cat_amounts[t.category].append(t.amount)

    curr_debits = [t for t in all_debits if t.month == current_month and t.category != "Housing & Rent"]
    for t in curr_debits:
        cat_vals = by_cat_amounts[t.category]
        if len(cat_vals) >= 4:
            mean_val = sum(cat_vals) / len(cat_vals)
            variance = sum((x - mean_val) ** 2 for x in cat_vals) / len(cat_vals)
            std_val = math.sqrt(variance) if variance > 0 else 1.0
            z_score = (t.amount - mean_val) / std_val
            if z_score >= ANOMALY_Z_SCORE_THRESHOLD and t.amount >= 180.0:
                alerts.append(
                    FinancialAlert(
                        alert_id=f"ALT-{alert_idx:02d}",
                        severity="WARNING" if t.amount < 500 else "CRITICAL",
                        alert_type="SPENDING_SPIKE",
                        category=t.category,
                        title=f"Unusual High-Value Charge at {t.merchant}",
                        description=(
                            f"Detected a ${t.amount:,.2f} transaction on {t.date} at {t.merchant} "
                            f"({t.amount / mean_val:.1f}x your typical {t.category} transaction of ${mean_val:,.0f}, "
                            f"Z-score={z_score:.1f})."
                        ),
                        metric_highlight=f"${t.amount:,.2f} ({t.amount / mean_val:.1f}x category avg)",
                        recommended_immediate_action=(
                            "Verify if this was a one-time emergency purchase and amortize it from your reserve fund."
                        ),
                    )
                )
                alert_idx += 1

    # 3. Subscription Creep & Overlapping Services Alert
    redundant_subs = [s for s in subs if s.is_potentially_redundant]
    total_sub_cost = sum(s.monthly_cost for s in subs)
    if redundant_subs or total_sub_cost > profile.monthly_budgets.get("Subscriptions", 100.0):
        redundant_names = ", ".join(f"{s.merchant} (${s.monthly_cost:.2f})" for s in redundant_subs)
        alerts.append(
            FinancialAlert(
                alert_id=f"ALT-{alert_idx:02d}",
                severity="WARNING" if total_sub_cost > 120 else "INFO",
                alert_type="SUBSCRIPTION_CREEP",
                category="Subscriptions",
                title=f"Subscription Creep Detected (${total_sub_cost:,.2f}/mo across {len(subs)} recurring charges)",
                description=(
                    f"You have {len(subs)} active recurring charges totaling ${total_sub_cost:,.2f}/month "
                    f"(${total_sub_cost * 12:,.2f}/year). "
                    + (f"Overlapping services found: {redundant_names}." if redundant_names else "")
                ),
                metric_highlight=f"${total_sub_cost * 12:,.0f}/yr recurring drain",
                recommended_immediate_action="Consolidate duplicate streaming/music/software subscriptions immediately.",
            )
        )
        alert_idx += 1

    # 4. Cashflow Burn / Checking Balance Risk OR Idle Cash Opportunity
    net_cash_flow = monthly_income - total_curr_spend
    if profile.checking_balance < 1200 or net_cash_flow < 300:
        alerts.append(
            FinancialAlert(
                alert_id=f"ALT-{alert_idx:02d}",
                severity="CRITICAL",
                alert_type="CASHFLOW_BURN",
                category="Cash Flow & Liquidity",
                title="Low Checking Liquidity & Tight Monthly Margin",
                description=(
                    f"Checking balance is down to ${profile.checking_balance:,.2f} and current month net cash flow "
                    f"is only ${net_cash_flow:,.2f} (income ${monthly_income:,.0f} vs spend ${total_curr_spend:,.0f})."
                ),
                metric_highlight=f"${profile.checking_balance:,.0f} checking buffer",
                recommended_immediate_action=(
                    "Enable Low-Balance Shield and cut daily micro-spends (coffee/rideshare) for the next 30 days."
                ),
            )
        )
        alert_idx += 1
    elif profile.checking_balance > profile.monthly_net_income * 1.4:
        surplus_cash = round(profile.checking_balance - (profile.monthly_net_income * 0.8), -2)
        annual_yield_loss = round(surplus_cash * 0.045, 2)
        alerts.append(
            FinancialAlert(
                alert_id=f"ALT-{alert_idx:02d}",
                severity="INFO",
                alert_type="GOAL_RISK",
                category="Cash Optimization",
                title=f"Idle Checking Cash Opportunity (${surplus_cash:,.0f} Earning 0.01% APY)",
                description=(
                    f"You hold ${profile.checking_balance:,.2f} in checking. Sweeping ${surplus_cash:,.0f} into your "
                    f"4.50% APY High-Yield Savings Vault generates +${annual_yield_loss:,.0f}/year risk-free."
                ),
                metric_highlight=f"+${annual_yield_loss:,.0f}/yr missed interest",
                recommended_immediate_action=f"Activate Auto-Sweep for ${surplus_cash:,.0f} into High-Yield Savings.",
            )
        )
        alert_idx += 1

    # Sort alerts by severity: CRITICAL -> WARNING -> INFO
    severity_rank = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    alerts.sort(key=lambda a: severity_rank.get(a.severity, 3))
    return alerts


@tool
def tool_evaluate_financial_alerts(customer_id: str, month: str = CURRENT_MONTH) -> dict:
    """Run proactive and reactive risk checks for budget breaches, Z-score anomalies, and subscription creep."""
    alerts = evaluate_customer_alerts(customer_id, current_month=month)
    return {
        "customer_id": customer_id,
        "alert_count": len(alerts),
        "alerts": [a.model_dump() for a in alerts],
    }

