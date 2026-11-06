"""
Agent 4: Alerting Agent
Monitors customer transaction streams both proactively (when opening the chatbot,
replacing static transaction notifications) and reactively during conversations.
Flags budget breaches, statistical spending spikes (Z-score outliers), subscription creep,
and cash-flow burn risks.
"""

from datetime import datetime
import time
from typing import Tuple
from src.config import CURRENT_MONTH
from src.llm.engine import llm_engine
from src.schemas.models import AgentTraceStep, AlertReport
from src.tools.anomaly_tools import evaluate_customer_alerts


class AlertingAgent:
    """Specialized agent for proactive overspending alerts and anomaly detection."""

    name = "Alerting Agent"

    def run(
        self,
        customer_id: str,
        step_order: int = 4,
    ) -> Tuple[AlertReport, AgentTraceStep]:
        start_ts = time.perf_counter()

        alerts = evaluate_customer_alerts(customer_id=customer_id, current_month=CURRENT_MONTH)
        crit_cnt = sum(1 for a in alerts if a.severity == "CRITICAL")
        warn_cnt = sum(1 for a in alerts if a.severity == "WARNING")
        info_cnt = sum(1 for a in alerts if a.severity == "INFO")

        if crit_cnt >= 1:
            risk_level = "HIGH"
        elif warn_cnt >= 1:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        top_alert_titles = "; ".join(f"[{a.severity}] {a.title}" for a in alerts[:3])
        default_headline = (
            f"Proactive Monitor detected {len(alerts)} active signals "
            f"({crit_cnt} Critical, {warn_cnt} Warning): {top_alert_titles}"
            if alerts
            else "All category budgets and transaction patterns are within healthy limits."
        )

        llm_headline, mode_used = llm_engine.generate_text(
            system_prompt=(
                "You are the Proactive Alerting Agent for a bank. Write a single urgent but helpful "
                "one-sentence alert banner summarizing the most important overspending or anomaly risk."
            ),
            user_prompt=f"Customer {customer_id} alerts:\n" + "\n".join(f"- {a.severity}: {a.title} ({a.description})" for a in alerts),
        )

        report = AlertReport(
            customer_id=customer_id,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            overall_risk_level=risk_level,
            critical_count=crit_cnt,
            warning_count=warn_cnt,
            info_count=info_cnt,
            alerts=alerts,
            proactive_headline=llm_headline or default_headline,
        )

        elapsed_ms = round((time.perf_counter() - start_ts) * 1000, 1)
        trace = AgentTraceStep(
            step_order=step_order,
            agent_name=self.name,
            action_type="TOOL_CALL",
            summary=(
                f"Executed anomaly & budget threshold scan: Risk Level={risk_level} "
                f"({crit_cnt} Critical, {warn_cnt} Warning, {info_cnt} Info alerts)."
            ),
            tools_invoked=["tool_evaluate_financial_alerts"],
            input_summary=f"customer_id={customer_id}, month={CURRENT_MONTH}",
            output_summary=f"Produced AlertReport with {len(alerts)} graded alerts.",
            latency_ms=elapsed_ms,
            llm_mode_used=mode_used,
        )

        return report, trace

