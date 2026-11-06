"""
Dual-Tier Memory Manager for the Personal Finance Advisor:
1. Short-Term Memory (STM): Sliding window of conversation turns + cached active analysis state
   per customer session, enabling multi-turn contextual follow-ups.
2. Long-Term Memory (LTM): Persistent JSON store tracking learned user facts, life events,
   custom budget preferences, and past advisory insights across sessions.
"""

from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from src.config import MEMORY_PATH
from src.data.db import db


class MemoryManager:
    """Manages both Short-Term conversational memory and Long-Term customer memory."""

    def __init__(self, memory_path: Path = MEMORY_PATH, max_stm_turns: int = 12):
        self.memory_path = memory_path
        self.max_stm_turns = max_stm_turns
        # Short-term session memory keyed by customer_id
        self._stm_history: Dict[str, List[Dict[str, str]]] = {}
        self._stm_cached_reports: Dict[str, Dict[str, Any]] = {}
        self._ensure_ltm_file()

    def _ensure_ltm_file(self) -> None:
        if not self.memory_path.exists():
            initial_ltm = {
                "CUST-001": {
                    "learned_facts": [
                        "Prefers keeping gym membership (CloudGym Pro) active despite high cost.",
                        "Trying to cook at home on weeknights to accelerate house down payment goal.",
                    ],
                    "past_advisory_notes": [
                        "2026-09-30: Flagged rising food delivery trend; user requested weekly nudges.",
                    ],
                    "custom_preferences": {"preferred_savings_day": "1st and 15th (Payday)"},
                },
                "CUST-002": {
                    "learned_facts": [
                        "Prioritizes kids' education (Kumon) and family museum activities.",
                        "Interested in low-risk, FDIC-insured high-yield savings options.",
                    ],
                    "past_advisory_notes": [
                        "2026-09-30: Grocery budget well maintained within $850 target.",
                    ],
                    "custom_preferences": {"preferred_savings_day": "2nd of month"},
                },
                "CUST-003": {
                    "learned_facts": [
                        "Freelance client payouts arrive irregularly between the 5th and 22nd.",
                        "Adobe CC and Figma are essential work tools (tax-deductible business expenses).",
                    ],
                    "past_advisory_notes": [
                        "2026-09-30: Recommended building a 2-month income smoothing buffer.",
                    ],
                    "custom_preferences": {"cash_buffer_floor": "$1,000"},
                },
            }
            self.memory_path.write_text(json.dumps(initial_ltm, indent=2))

    def _read_ltm(self) -> Dict[str, Any]:
        try:
            return json.loads(self.memory_path.read_text())
        except Exception:
            self._ensure_ltm_file()
            return json.loads(self.memory_path.read_text())

    def _write_ltm(self, data: Dict[str, Any]) -> None:
        self.memory_path.write_text(json.dumps(data, indent=2))

    # --- Short-Term Memory Operations ---

    def get_chat_history(self, customer_id: str) -> List[Dict[str, str]]:
        """Return recent conversation turns for the customer."""
        return list(self._stm_history.get(customer_id, []))

    def add_chat_turn(self, customer_id: str, role: str, content: str) -> None:
        """Append a message to short-term session memory and keep within window."""
        if customer_id not in self._stm_history:
            self._stm_history[customer_id] = []
        self._stm_history[customer_id].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
            }
        )
        if len(self._stm_history[customer_id]) > self.max_stm_turns:
            self._stm_history[customer_id] = self._stm_history[customer_id][-self.max_stm_turns :]

    def cache_session_artifacts(
        self,
        customer_id: str,
        spending_report: Optional[Dict[str, Any]] = None,
        recommendation_report: Optional[Dict[str, Any]] = None,
        alert_report: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cache latest structured agent outputs in STM for quick follow-up reference."""
        if customer_id not in self._stm_cached_reports:
            self._stm_cached_reports[customer_id] = {}
        if spending_report:
            self._stm_cached_reports[customer_id]["spending_report"] = spending_report
        if recommendation_report:
            self._stm_cached_reports[customer_id]["recommendation_report"] = recommendation_report
        if alert_report:
            self._stm_cached_reports[customer_id]["alert_report"] = alert_report

    def get_cached_artifacts(self, customer_id: str) -> Dict[str, Any]:
        return self._stm_cached_reports.get(customer_id, {})

    def clear_session(self, customer_id: str) -> None:
        """Clear short-term session history for a customer."""
        self._stm_history[customer_id] = []
        self._stm_cached_reports[customer_id] = {}

    # --- Long-Term Memory Operations ---

    def get_long_term_memory(self, customer_id: str) -> Dict[str, Any]:
        """Retrieve persistent customer facts, preferences, and profile summary."""
        ltm = self._read_ltm()
        cust_ltm = ltm.get(
            customer_id,
            {"learned_facts": [], "past_advisory_notes": [], "custom_preferences": {}},
        )
        profile = db.get_profile(customer_id)
        return {
            "customer_id": customer_id,
            "name": profile.name,
            "persona_tag": profile.persona_tag,
            "monthly_net_income": profile.monthly_net_income,
            "checking_balance": profile.checking_balance,
            "savings_balance": profile.savings_balance,
            "monthly_budgets": profile.monthly_budgets,
            "financial_goals": [g.model_dump() for g in profile.financial_goals],
            "risk_tolerance": profile.risk_tolerance,
            "learned_facts": cust_ltm.get("learned_facts", []),
            "past_advisory_notes": cust_ltm.get("past_advisory_notes", []),
            "custom_preferences": cust_ltm.get("custom_preferences", {}),
        }

    def record_learned_fact(self, customer_id: str, fact: str) -> None:
        """Persist a new fact or preference learned during conversation into Long-Term Memory."""
        ltm = self._read_ltm()
        if customer_id not in ltm:
            ltm[customer_id] = {"learned_facts": [], "past_advisory_notes": [], "custom_preferences": {}}
        if fact and fact not in ltm[customer_id]["learned_facts"]:
            ltm[customer_id]["learned_facts"].append(fact)
            self._write_ltm(ltm)

    def extract_and_persist_facts_from_query(self, customer_id: str, user_query: str) -> List[str]:
        """
        Detect explicit user statements about goals, budgets, or life events in the query
        and automatically persist them into Long-Term Memory + CustomerProfile if applicable.
        """
        extracted: List[str] = []
        q_lower = user_query.lower()

        # Check for new savings goal or target amount statements
        if any(kw in q_lower for kw in ["i am saving for", "i'm saving for", "my goal is", "want to save", "keep my", "don't cancel", "do not cancel", "wedding", "vacation", "car", "trip"]):
            fact_str = f"[{datetime.now().strftime('%Y-%m-%d')}] User stated: \"{user_query.strip()}\""
            self.record_learned_fact(customer_id, fact_str)
            extracted.append(fact_str)

        # Check if user asks to update a specific category budget: e.g. "set my dining budget to $500"
        budget_match = re.search(
            r"(?:set|change|update)\s+(?:my\s+)?(food|dining|entertainment|shopping|groceries|subscriptions)\s+budget\s+to\s+\$?(\d+(?:\.\d+)?)",
            q_lower,
        )
        if budget_match:
            cat_raw, new_amt_str = budget_match.group(1), budget_match.group(2)
            cat_map = {
                "food": "Food & Dining",
                "dining": "Food & Dining",
                "entertainment": "Entertainment",
                "shopping": "Shopping",
                "groceries": "Groceries",
                "subscriptions": "Subscriptions",
            }
            mapped_cat = cat_map.get(cat_raw, "Food & Dining")
            new_amt = float(new_amt_str)
            profile = db.get_profile(customer_id)
            old_amt = profile.monthly_budgets.get(mapped_cat, 0.0)
            profile.monthly_budgets[mapped_cat] = new_amt
            db.update_profile(profile)
            note = f"Updated {mapped_cat} monthly budget from ${old_amt:,.0f} to ${new_amt:,.0f}."
            self.record_learned_fact(customer_id, note)
            extracted.append(note)

        return extracted


# Singleton memory manager
memory_manager = MemoryManager()

