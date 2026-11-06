"""
Deterministic transaction aggregation and Month-over-Month (MoM) analysis tools
used by the Spending Analysis Agent.
All arithmetic is computed in Python rather than delegated to LLM token prediction.
"""

from collections import defaultdict
from typing import Dict, List, Tuple
from langchain_core.tools import tool
from src.config import CURRENT_MONTH, PREVIOUS_MONTH, TWO_MONTHS_AGO
from src.data.db import db
from src.schemas.models import (
    CategorySpendSummary,
    MicroSpendPattern,
    RecurringSubscription,
)


def compute_category_breakdown(
    customer_id: str,
    current_month: str = CURRENT_MONTH,
    previous_month: str = PREVIOUS_MONTH,
) -> Tuple[List[CategorySpendSummary], float, float, float]:
    """
    Calculate category-level spending, MoM deltas, 3-month averages,
    and budget utilization percentages for a customer.
    """
    profile = db.get_profile(customer_id)
    all_debits = db.query_transactions(customer_id, transaction_type="debit")
    all_credits = db.query_transactions(customer_id, month=current_month, transaction_type="credit")

    monthly_income = sum(t.amount for t in all_credits) or profile.monthly_net_income

    # Group by (month, category)
    by_month_cat: Dict[Tuple[str, str], List] = defaultdict(list)
    categories_seen = set(profile.monthly_budgets.keys())

    for t in all_debits:
        by_month_cat[(t.month, t.category)].append(t)
        categories_seen.add(t.category)

    total_current_spend = sum(t.amount for t in all_debits if t.month == current_month)
    total_prev_spend = sum(t.amount for t in all_debits if t.month == previous_month)

    summaries: List[CategorySpendSummary] = []
    for cat in sorted(categories_seen):
        curr_txns = by_month_cat.get((current_month, cat), [])
        prev_txns = by_month_cat.get((previous_month, cat), [])
        aug_txns = by_month_cat.get((TWO_MONTHS_AGO, cat), [])

        curr_amt = round(sum(t.amount for t in curr_txns), 2)
        prev_amt = round(sum(t.amount for t in prev_txns), 2)
        aug_amt = round(sum(t.amount for t in aug_txns), 2)

        three_mo_avg = round((curr_amt + prev_amt + aug_amt) / 3.0, 2)
        budget = profile.monthly_budgets.get(cat, round(three_mo_avg * 1.1, 2) or 200.0)
        utilization_pct = round((curr_amt / budget) * 100.0, 1) if budget > 0 else 0.0

        mom_delta = round(curr_amt - prev_amt, 2)
        mom_pct = round((mom_delta / prev_amt) * 100.0, 1) if prev_amt > 0 else (100.0 if curr_amt > 0 else 0.0)
        share_pct = round((curr_amt / total_current_spend) * 100.0, 1) if total_current_spend > 0 else 0.0

        # Top merchants in current month
        merchant_totals: Dict[str, float] = defaultdict(float)
        for t in curr_txns:
            merchant_totals[t.merchant] += t.amount
        top_merchants = [
            f"`{m} (${amt:,.0f})`"
            for m, amt in sorted(merchant_totals.items(), key=lambda x: x[1], reverse=True)[:2]
        ]

        summaries.append(
            CategorySpendSummary(
                category=cat,
                current_month_spend=curr_amt,
                previous_month_spend=prev_amt,
                three_month_avg_spend=three_mo_avg,
                monthly_budget=budget,
                budget_utilization_pct=utilization_pct,
                mom_change_amount=mom_delta,
                mom_change_pct=mom_pct,
                share_of_total_spend_pct=share_pct,
                transaction_count=len(curr_txns),
                top_merchants=top_merchants,
            )
        )

    # Sort by highest current month spend first
    summaries.sort(key=lambda s: s.current_month_spend, reverse=True)
    return summaries, round(total_current_spend, 2), round(total_prev_spend, 2), round(monthly_income, 2)


def detect_subscriptions_and_micro_spends(
    customer_id: str,
    current_month: str = CURRENT_MONTH,
) -> Tuple[List[RecurringSubscription], List[MicroSpendPattern]]:
    """
    Identify recurring subscription charges (including overlapping/redundant subscriptions)
    and high-frequency micro-spend habits (e.g., food delivery, coffee shops, rideshares).
    """
    all_debits = db.query_transactions(customer_id, transaction_type="debit")
    curr_debits = [t for t in all_debits if t.month == current_month]

    # 1. Recurring Subscriptions in current month
    sub_txns = [t for t in curr_debits if t.category == "Subscriptions" or (t.is_recurring and t.category not in ("Housing & Rent", "Utilities", "Transportation"))]

    # Count subcategory occurrences to flag overlapping services (e.g., multiple Video Streaming or Music Streaming)
    subcat_counts: Dict[str, int] = defaultdict(int)
    for t in sub_txns:
        subcat_counts[t.subcategory] += 1

    # Count total appearances across 3 months per merchant
    merchant_occurrences: Dict[str, int] = defaultdict(int)
    for t in all_debits:
        merchant_occurrences[t.merchant] += 1

    subscriptions: List[RecurringSubscription] = []
    for t in sorted(sub_txns, key=lambda x: x.amount, reverse=True):
        is_redundant = subcat_counts[t.subcategory] > 1
        reason = None
        if is_redundant:
            reason = f"Overlapping {t.subcategory} service ({subcat_counts[t.subcategory]} active subscriptions in this tier)"
        if t.notes and "Price increased" in t.notes:
            reason = (reason + " | " if reason else "") + t.notes

        subscriptions.append(
            RecurringSubscription(
                merchant=t.merchant,
                category=t.subcategory,
                monthly_cost=t.amount,
                annualized_cost=round(t.amount * 12, 2),
                occurrences=merchant_occurrences[t.merchant],
                is_potentially_redundant=is_redundant,
                redundancy_reason=reason,
            )
        )

    # 2. High-Frequency Micro-Spend / Habit Patterns in current month (3+ transactions in same subcategory)
    by_subcat: Dict[str, List] = defaultdict(list)
    for t in curr_debits:
        if t.subcategory in ("Food Delivery", "Coffee Shop", "Rideshare", "Fast Casual"):
            by_subcat[t.subcategory].append(t)

    micro_patterns: List[MicroSpendPattern] = []
    for subcat, txns in by_subcat.items():
        if len(txns) >= 3:
            total_amt = round(sum(x.amount for x in txns), 2)
            avg_ticket = round(total_amt / len(txns), 2)
            examples = list(dict.fromkeys(x.merchant for x in txns))[:3]
            micro_patterns.append(
                MicroSpendPattern(
                    pattern_name=f"Frequent {subcat} Habit",
                    category=txns[0].category,
                    merchant_examples=examples,
                    transaction_count_monthly=len(txns),
                    average_ticket=avg_ticket,
                    total_monthly_spend=total_amt,
                    annualized_impact=round(total_amt * 12, 2),
                )
            )

    micro_patterns.sort(key=lambda m: m.total_monthly_spend, reverse=True)
    return subscriptions, micro_patterns


# LangChain @tool wrappers for formal tool registry inspection & LLM binding
@tool
def tool_aggregate_spending_by_category(customer_id: str, month: str = CURRENT_MONTH) -> dict:
    """Aggregate customer transactions by category, calculating MoM deltas and budget utilization."""
    summaries, curr_total, prev_total, income = compute_category_breakdown(customer_id, current_month=month)
    return {
        "customer_id": customer_id,
        "month": month,
        "monthly_income": income,
        "total_current_spend": curr_total,
        "total_previous_spend": prev_total,
        "categories": [s.model_dump() for s in summaries],
    }


@tool
def tool_detect_subscriptions_and_habits(customer_id: str, month: str = CURRENT_MONTH) -> dict:
    """Detect recurring subscriptions, redundant memberships, and high-frequency micro-spend habits."""
    subs, micro = detect_subscriptions_and_micro_spends(customer_id, current_month=month)
    return {
        "subscriptions": [s.model_dump() for s in subs],
        "micro_spend_patterns": [m.model_dump() for m in micro],
    }

