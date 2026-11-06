"""
Agent 1: Planner Agent
Understands the user query, consults Short-Term & Long-Term Memory, constructs an
explicit ExecutionPlan for LangGraph routing, enforces the dependency that Recommendation
Agent always receives input from Spending Analysis Agent, and synthesizes the final response.
"""

import re
import time
from typing import Any, Dict, List, Optional, Tuple
from src.llm.engine import llm_engine
from src.memory.memory_manager import memory_manager
from src.schemas.models import (
    AgentTraceStep,
    AlertReport,
    ExecutionPlan,
    RecommendationReport,
    SpendingAnalysisReport,
)
from src.tools.guardrail_tools import (
    evaluate_domain_guardrail,
    format_guardrail_refusal,
)


class PlannerAgent:
    """Front-door orchestrator, intent planner, and final response synthesizer."""

    name = "Planner Agent"

    def create_plan(
        self,
        customer_id: str,
        user_query: str,
        memory_context: Dict[str, Any],
    ) -> Tuple[ExecutionPlan, AgentTraceStep]:
        """
        1. Run Dual-Mode Domain Guardrail (LLM Intent Guardrail if API key is present,
           or Deterministic Pattern & Domain Guardrail if no API key is set).
        2. If allowed, analyze user query + memory to decide which specialized agents and tools to invoke.
        """
        start_ts = time.perf_counter()

        # Step 0: Domain & Safety Guardrail Evaluation
        is_allowed, guardrail_reason, guardrail_mode = evaluate_domain_guardrail(user_query)
        if not is_allowed:
            plan = ExecutionPlan(
                user_intent="BLOCKED_OFF_TOPIC",
                is_allowed=False,
                guardrail_reason=guardrail_reason,
                focus_categories=[],
                target_savings_override=None,
                run_spending_agent=False,
                run_recommendation_agent=False,
                run_alerting_agent=False,
                reasoning=f"Guardrail intercepted off-topic/unsafe query: {guardrail_reason}",
                planned_steps=[
                    "1. Evaluate query against Personal Finance & Banking Guardrail -> BLOCKED.",
                    "2. Skip Spending Analysis, Recommendation, and Alerting sub-agents.",
                    "3. Return polite domain-scope refusal and log guardrail event.",
                ],
                extracted_user_facts=[],
            )
            elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
            trace = AgentTraceStep(
                step_order=1,
                agent_name=self.name,
                action_type="GUARDRAIL",
                summary=f"BLOCKED off-topic query: {guardrail_reason}",
                tools_invoked=["evaluate_domain_guardrail"],
                input_summary=f"query='{user_query[:80]}', customer_id={customer_id}",
                output_summary="ExecutionPlan(intent=BLOCKED_OFF_TOPIC, is_allowed=False)",
                latency_ms=elapsed_ms,
                llm_mode_used=guardrail_mode,
            )
            return plan, trace

        q = user_query.lower()

        # Extract and persist any new user facts or budget updates into Long-Term Memory
        extracted_facts = memory_manager.extract_and_persist_facts_from_query(customer_id, user_query)

        # Detect category filters mentioned in query
        category_keywords = {
            "Food & Dining": ["food", "dining", "restaurant", "doordash", "ubereats", "delivery", "coffee", "eat"],
            "Subscriptions": ["subscription", "recurring", "streaming", "netflix", "streamflix", "spotify", "membership"],
            "Entertainment": ["entertainment", "concert", "movie", "nightlife", "fun"],
            "Groceries": ["grocery", "groceries", "supermarket", "whole foods", "trader joe"],
            "Utilities": ["utility", "utilities", "electric", "power", "internet", "bill"],
            "Shopping": ["shopping", "amazon", "clothes", "electronics", "appliance", "homedepot"],
            "Transportation": ["transport", "uber", "lyft", "rideshare", "gas", "fuel", "transit"],
        }
        focus_cats = [
            cat for cat, kws in category_keywords.items() if any(kw in q for kw in kws)
        ]

        # Check for custom dollar target in query (e.g., "save $400")
        amt_match = re.search(r"\$(\d+(?:,\d{3})*(?:\.\d+)?)", user_query)
        target_override = float(amt_match.group(1).replace(",", "")) if amt_match else None

        # Classify intent & determine sub-agent invocation plan
        wants_alerts_only = any(w in q for w in ["alert", "anomaly", "anomalies", "unusual", "spike", "warning", "breach", "overspend"]) and not any(
            w in q for w in ["save", "recommend", "optimize", "advice", "how can i", "plan"]
        )
        wants_spending_only = any(w in q for w in ["breakdown", "categor", "how much did i spend", "where did my money", "summary", "transactions"]) and not any(
            w in q for w in ["save", "recommend", "optimize", "advice", "reduce", "cut", "budget"]
        )
        wants_recommendations = any(
            w in q for w in ["save", "recommend", "optimize", "advice", "reduce", "cut", "budget", "goal", "improve", "help", "what should i"]
        )

        if wants_alerts_only and not wants_recommendations:
            intent = "ALERTS_AND_ANOMALIES"
            run_spending = True   # Context for alerts
            run_rec = False
            run_alert = True
            steps = [
                "1. Consult Long-Term Memory for customer risk profile & budget thresholds.",
                "2. Dispatch Spending Analysis Agent to compute category baselines and MoM shifts.",
                "3. Dispatch Alerting Agent to scan for budget breaches, Z-score outliers, and subscription creep.",
                "4. Synthesize risk report with immediate protective actions.",
            ]
        elif wants_spending_only and not wants_recommendations:
            intent = "SPENDING_ANALYSIS_ONLY"
            run_spending = True
            run_rec = False
            run_alert = True
            steps = [
                "1. Retrieve customer profile and monthly category budgets from Memory.",
                "2. Dispatch Spending Analysis Agent to categorize transactions, compute MoM deltas, and detect habits.",
                "3. Dispatch Alerting Agent to highlight any active category breaches.",
                "4. Collate structured spending breakdown for the customer.",
            ]
        elif wants_recommendations:
            intent = "SAVINGS_AND_RECOMMENDATIONS"
            # CRITICAL ARCHITECTURAL RULE: Recommendation Agent MUST get its input from Spending Analysis Agent
            run_spending = True
            run_rec = True
            run_alert = True
            steps = [
                "1. Load customer financial goals and learned preferences from Long-Term Memory.",
                "2. Execute Spending Analysis Agent FIRST to generate structured `SpendingAnalysisReport`.",
                "3. Pass `SpendingAnalysisReport` directly into Recommendation Agent to simulate savings & budget optimizations.",
                "4. Run Alerting Agent to cross-check urgent overspending risks.",
                "5. Synthesize unified financial advisory response.",
            ]
        else:
            intent = "FULL_ADVISORY"
            run_spending = True
            run_rec = True
            run_alert = True
            steps = [
                "1. Load customer profile, budgets, and conversation context from Dual-Tier Memory.",
                "2. Execute Spending Analysis Agent to analyze 3-month transaction patterns.",
                "3. Pipe `SpendingAnalysisReport` output into Recommendation Agent for personalized optimization.",
                "4. Execute Alerting Agent for proactive overspending & anomaly checks.",
                "5. Collate all multi-agent findings into an actionable advisory response.",
            ]

        plan = ExecutionPlan(
            user_intent=intent,
            focus_categories=focus_cats,
            target_savings_override=target_override,
            run_spending_agent=run_spending,
            run_recommendation_agent=run_rec,
            run_alerting_agent=run_alert,
            reasoning=(
                f"Classified intent as '{intent}'. "
                + (
                    "Enforcing mandatory pipeline dependency: Spending Analysis Agent -> Recommendation Agent."
                    if run_rec
                    else "Routing to Spending Analysis & Alerting agents."
                )
            ),
            planned_steps=steps,
            extracted_user_facts=extracted_facts,
        )

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
        trace = AgentTraceStep(
            step_order=1,
            agent_name=self.name,
            action_type="PLANNING",
            summary=(
                f"Parsed query & memory -> Intent={intent}, Focus={focus_cats or ['All Categories']}. "
                f"Planned pipeline: Spending={run_spending} -> Recommendation={run_rec}, Alerting={run_alert}."
                + (f" Persisted {len(extracted_facts)} new memory fact(s)." if extracted_facts else "")
            ),
            tools_invoked=["memory_manager.get_long_term_memory", "memory_manager.extract_and_persist_facts"],
            input_summary=f"query='{user_query[:80]}', customer_id={customer_id}",
            output_summary=f"ExecutionPlan(intent={intent}, steps={len(steps)})",
            latency_ms=elapsed_ms,
            llm_mode_used="Deterministic Intent & Dependency Planner",
        )

        return plan, trace

    def synthesize_response(
        self,
        customer_id: str,
        user_query: str,
        memory_context: Dict[str, Any],
        plan: ExecutionPlan,
        spending_report: Optional[SpendingAnalysisReport],
        recommendation_report: Optional[RecommendationReport],
        alert_report: Optional[AlertReport],
        chat_history: List[Dict[str, str]],
        step_order: int = 5,
    ) -> Tuple[str, AgentTraceStep]:
        """Collate outputs from all executed agents into a cohesive, conversational response."""
        start_ts = time.perf_counter()
        customer_name = memory_context.get("name", "Valued Customer")
        first_name = customer_name.split()[0]

        # Handle Guardrail Refusal cleanly without invoking sub-agent synthesis
        if not plan.is_allowed or plan.user_intent == "BLOCKED_OFF_TOPIC":
            refusal_md = format_guardrail_refusal(
                user_query=user_query,
                reason=plan.guardrail_reason or "Query is outside personal finance & banking scope.",
                customer_name=customer_name,
            )
            elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
            trace = AgentTraceStep(
                step_order=step_order,
                agent_name=self.name,
                action_type="GUARDRAIL",
                summary="Returned compliant banking scope refusal without executing financial tools or sub-agents.",
                tools_invoked=["format_guardrail_refusal"],
                input_summary=f"blocked_query='{user_query[:60]}'",
                output_summary="Polite guardrail refusal message returned to user.",
                latency_ms=elapsed_ms,
                llm_mode_used="Guardrail Refusal Synthesizer",
            )
            return refusal_md, trace

        # Build structured markdown sections deterministically first (guarantees 100% numeric accuracy)
        sections: List[str] = []

        # Greeting & Memory Acknowledgment
        if plan.extracted_user_facts:
            sections.append(
                f"> **Memory Updated:** I've saved this to your financial profile: *\"{plan.extracted_user_facts[-1]}\"*\n"
            )

        # 1. Spending Analysis Section
        if spending_report:
            sections.append(
                f"### 1. Spending Analysis ({spending_report.current_month} vs. {spending_report.previous_month})\n"
                f"Hi **{first_name}**, here is what the **Spending Analysis Agent** uncovered across your accounts:\n"
                f"- **Monthly Net Income:** `${spending_report.monthly_income:,.2f}` | "
                f"**Total Spend:** `${spending_report.total_current_month_spend:,.2f}` "
                f"(`{spending_report.mom_total_change_pct:+.1f}%` MoM)\n"
                f"- **Net Cash Flow:** `${spending_report.net_cash_flow:,.2f}` "
                f"(Current Savings Rate: **`{spending_report.savings_rate_pct:.1f}%`**)\n"
            )

            # Filter categories if user asked about specific ones, else show top 6
            cats_to_show = spending_report.category_breakdown
            if plan.focus_categories:
                focused = [c for c in cats_to_show if c.category in plan.focus_categories]
                if focused:
                    cats_to_show = focused + [c for c in cats_to_show if c.category not in plan.focus_categories][:3]
                else:
                    cats_to_show = cats_to_show[:6]
            else:
                cats_to_show = cats_to_show[:6]

            table_lines = [
                "| Category | Current Spend | Budget | Utilization | MoM Change | Top Merchants |",
                "| :--- | :---: | :---: | :---: | :---: | :--- |",
            ]
            for c in cats_to_show:
                status_icon = "🔴" if c.budget_utilization_pct >= 100 and c.category != "Housing & Rent" else ("🟡" if c.budget_utilization_pct >= 85 and c.category != "Housing & Rent" else "🟢")
                merchants_str = ", ".join(c.top_merchants[:2]) if c.top_merchants else "—"
                table_lines.append(
                    f"| **{c.category}** | `${c.current_month_spend:,.2f}` | `${c.monthly_budget:,.0f}` | "
                    f"{status_icon} `{c.budget_utilization_pct:.0f}%` | `{c.mom_change_pct:+.1f}%` | {merchants_str} |"
                )
            sections.append("\n".join(table_lines))

            # Highlight micro-spends & subscriptions
            pattern_bullets = []
            for mp in spending_report.micro_spend_patterns:
                pattern_bullets.append(
                    f"- **High-Frequency Habit ({mp.pattern_name}):** `{mp.transaction_count_monthly}x` transactions "
                    f"(avg `${mp.average_ticket:.2f}`) totaling **`${mp.total_monthly_spend:,.2f}/mo`** (`${mp.annualized_impact:,.0f}/yr`)."
                )
            if spending_report.recurring_subscriptions:
                redundant = [s for s in spending_report.recurring_subscriptions if s.is_potentially_redundant]
                pattern_bullets.append(
                    f"- **Recurring Subscriptions (`{len(spending_report.recurring_subscriptions)}` active):** "
                    f"**`${spending_report.total_subscription_monthly_cost:,.2f}/mo`** (`${spending_report.total_subscription_monthly_cost * 12:,.0f}/yr`)"
                    + (
                        f" — includes `{len(redundant)}` overlapping services ({', '.join(s.merchant for s in redundant[:3])})."
                        if redundant
                        else "."
                    )
                )
            if pattern_bullets:
                sections.append("**Detected Behavioral Patterns:**\n" + "\n".join(pattern_bullets))

        # 2. Alerting Agent Section
        if alert_report and alert_report.alerts:
            alert_lines = [
                f"### 2. Proactive Risk & Overspending Alerts (`Risk Level: {alert_report.overall_risk_level}`)"
            ]
            for a in alert_report.alerts[:4]:
                badge = "🚨 **CRITICAL**" if a.severity == "CRITICAL" else ("⚠️ **WARNING**" if a.severity == "WARNING" else "ℹ️ **INFO**")
                alert_lines.append(
                    f"- {badge} — **{a.title}**: {a.description} *(Action: {a.recommended_immediate_action})*"
                )
            sections.append("\n".join(alert_lines))

        # 3. Recommendation Agent Section (derived from Spending Analysis Agent)
        if recommendation_report and recommendation_report.recommendations:
            rec_lines = [
                "### 3. Personalized Savings & Budget Optimization Plan\n"
                "*Generated by the **Recommendation Agent** using the **Spending Analysis Agent's** output and your long-term goals:*\n"
                f"- **Total Unlockable Savings:** **`${recommendation_report.total_potential_monthly_savings:,.2f}/month`** "
                f"(**`${recommendation_report.total_potential_annual_savings:,.2f}/year`**)\n"
                f"- **Projected Savings Rate Boost:** `{spending_report.savings_rate_pct if spending_report else 0:.1f}%` → "
                f"**`{recommendation_report.projected_new_savings_rate_pct:.1f}%`**\n"
            ]
            for r in recommendation_report.recommendations:
                rec_lines.append(
                    f"1. **{r.title}** *(Save **`${r.estimated_monthly_savings:,.2f}/mo`** | **`${r.estimated_annual_savings:,.0f}/yr`)*\n"
                    f"   - *Why (from Spending Analysis):* {r.observation_from_spending_agent}\n"
                    f"   - *Recommended Action:* {r.proposed_action}\n"
                    f"   - *Goal Impact:* {r.goal_acceleration_impact}"
                )
            if recommendation_report.recommended_bank_features:
                rec_lines.append(
                    "\n**Smart Bank Automation You Can Activate Today:**\n"
                    + "\n".join(f"- {feat}" for feat in recommendation_report.recommended_bank_features[:2])
                )
            sections.append("\n".join(rec_lines))

        # Incorporate Long-Term Memory context note
        learned_facts = memory_context.get("learned_facts", [])
        if learned_facts:
            sections.append(
                f"---\n*💡 **Personalization Context (from Long-Term Memory):** Kept in mind your profile note: "
                f"\"{learned_facts[-1]}\"*"
            )

        deterministic_markdown = "\n\n".join(sections)

        # If live Gemini API key is configured, allow Gemini to polish the conversational framing
        # while strictly preserving all markdown tables and exact dollar numbers.
        llm_response, mode_used = llm_engine.generate_text(
            system_prompt=(
                "You are the Planner Agent for an enterprise bank's AI Personal Finance Advisor. "
                "Present the collated multi-agent report clearly and empathetically to the customer. "
                "Keep all markdown tables, exact dollar numbers, percentages, and explicit references "
                "to the Spending Analysis Agent, Recommendation Agent, and Alerting Agent intact."
            ),
            user_prompt=(
                f"Customer Name: {customer_name}\n"
                f"User Query: {user_query}\n"
                f"Recent Chat History: {chat_history[-2:] if chat_history else 'None'}\n"
                f"Structured Multi-Agent Report to Present:\n{deterministic_markdown}"
            ),
        )

        raw_text = llm_response or deterministic_markdown
        # Replace any unescaped $ outside code spans with \$ so Streamlit KaTeX never renders dollar amounts as ugly math italic
        final_text = self._escape_dollars_for_streamlit(raw_text)
        elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)

        trace = AgentTraceStep(
            step_order=step_order,
            agent_name=self.name,
            action_type="SYNTHESIS",
            summary=(
                "Collated outputs from Spending Analysis Agent, Recommendation Agent, and Alerting Agent "
                "with Long-Term Memory into final customer response."
            ),
            tools_invoked=["memory_manager.add_chat_turn", "memory_manager.cache_session_artifacts"],
            input_summary=(
                f"spending_report={'Yes' if spending_report else 'No'}, "
                f"recommendation_report={'Yes' if recommendation_report else 'No'}, "
                f"alert_report={'Yes' if alert_report else 'No'}"
            ),
            output_summary=f"Generated {len(final_text)} chars markdown advisory response.",
            latency_ms=elapsed_ms,
            llm_mode_used=mode_used,
        )

        return final_text, trace

    @staticmethod
    def _escape_dollars_for_streamlit(text: str) -> str:
        """
        Escape literal '$' signs outside of backtick code spans (`...`) so that
        Streamlit's Markdown/KaTeX renderer does not treat two dollar amounts in the
        same sentence (e.g. '$742 ... $450') as a LaTeX math block with italic font and uneven spaces.
        """
        parts = text.split("`")
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r"(?<!\\)\$", r"\$", parts[i])
        return "`".join(parts)

