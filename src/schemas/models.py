"""Pydantic data models and TypedDict state definitions for LangGraph orchestration."""

from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Represents a single bank transaction."""
    transaction_id: str
    customer_id: str
    date: str  # YYYY-MM-DD
    month: str  # YYYY-MM
    merchant: str
    category: str
    subcategory: str
    amount: float
    transaction_type: Literal["debit", "credit"] = "debit"
    is_recurring: bool = False
    notes: Optional[str] = None


class FinancialGoal(BaseModel):
    """Represents a customer's savings or debt reduction goal."""
    goal_id: str
    name: str
    target_amount: float
    current_amount: float
    target_date: str
    priority: Literal["High", "Medium", "Low"] = "High"


class CustomerProfile(BaseModel):
    """Represents a customer's static profile, budgets, and goals."""
    customer_id: str
    name: str
    persona_tag: str
    monthly_net_income: float
    checking_balance: float
    savings_balance: float
    monthly_budgets: Dict[str, float] = Field(default_factory=dict)
    financial_goals: List[FinancialGoal] = Field(default_factory=list)
    risk_tolerance: Literal["Conservative", "Moderate", "Aggressive"] = "Moderate"
    preferred_communication_style: str = "Concise, analytical, and encouraging"


class CategorySpendSummary(BaseModel):
    """Aggregated spending and MoM comparison for a single category."""
    category: str
    current_month_spend: float
    previous_month_spend: float
    three_month_avg_spend: float
    monthly_budget: float
    budget_utilization_pct: float
    mom_change_amount: float
    mom_change_pct: float
    share_of_total_spend_pct: float
    transaction_count: int
    top_merchants: List[str] = Field(default_factory=list)


class RecurringSubscription(BaseModel):
    """Detected recurring subscription or regular membership charge."""
    merchant: str
    category: str
    monthly_cost: float
    annualized_cost: float
    occurrences: int
    is_potentially_redundant: bool = False
    redundancy_reason: Optional[str] = None


class MicroSpendPattern(BaseModel):
    """High-frequency small transactions that add up significantly."""
    pattern_name: str
    category: str
    merchant_examples: List[str]
    transaction_count_monthly: int
    average_ticket: float
    total_monthly_spend: float
    annualized_impact: float


class SpendingAnalysisReport(BaseModel):
    """Structured output produced by the Spending Analysis Agent and passed to the Recommendation Agent."""
    customer_id: str
    analysis_period: str
    current_month: str
    previous_month: str
    monthly_income: float
    total_current_month_spend: float
    total_previous_month_spend: float
    net_cash_flow: float
    savings_rate_pct: float
    mom_total_change_pct: float
    category_breakdown: List[CategorySpendSummary] = Field(default_factory=list)
    top_overspent_categories: List[str] = Field(default_factory=list)
    fastest_growing_categories: List[str] = Field(default_factory=list)
    recurring_subscriptions: List[RecurringSubscription] = Field(default_factory=list)
    total_subscription_monthly_cost: float = 0.0
    micro_spend_patterns: List[MicroSpendPattern] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    executive_summary: str = ""


class ActionableRecommendation(BaseModel):
    """A single quantified financial recommendation."""
    rec_id: str
    title: str
    target_category: str
    observation_from_spending_agent: str
    proposed_action: str
    estimated_monthly_savings: float
    estimated_annual_savings: float
    goal_acceleration_impact: str
    difficulty: Literal["Easy", "Medium", "Hard"] = "Easy"
    priority: int = 1


class BudgetOptimizationRow(BaseModel):
    """Proposed budget adjustment for a category."""
    category: str
    current_spend: float
    current_budget: float
    recommended_budget: float
    monthly_delta: float
    rationale: str


class RecommendationReport(BaseModel):
    """Structured output from the Recommendation Agent (strictly derived from SpendingAnalysisReport)."""
    customer_id: str
    source_analysis_period: str
    total_potential_monthly_savings: float
    total_potential_annual_savings: float
    projected_new_savings_rate_pct: float
    recommendations: List[ActionableRecommendation] = Field(default_factory=list)
    budget_optimizations: List[BudgetOptimizationRow] = Field(default_factory=list)
    recommended_bank_features: List[str] = Field(default_factory=list)
    strategic_summary: str = ""


class FinancialAlert(BaseModel):
    """A single proactive or reactive financial alert."""
    alert_id: str
    severity: Literal["CRITICAL", "WARNING", "INFO"]
    alert_type: Literal["BUDGET_BREACH", "SPENDING_SPIKE", "SUBSCRIPTION_CREEP", "CASHFLOW_BURN", "GOAL_RISK"]
    category: str
    title: str
    description: str
    metric_highlight: str
    recommended_immediate_action: str


class AlertReport(BaseModel):
    """Structured output from the Alerting Agent."""
    customer_id: str
    generated_at: str
    overall_risk_level: Literal["HIGH", "MODERATE", "LOW"]
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    alerts: List[FinancialAlert] = Field(default_factory=list)
    proactive_headline: str = ""


class ExecutionPlan(BaseModel):
    """Structured plan created by the Planner Agent before invoking sub-agents."""
    user_intent: Literal[
        "FULL_ADVISORY",
        "SPENDING_ANALYSIS_ONLY",
        "SAVINGS_AND_RECOMMENDATIONS",
        "ALERTS_AND_ANOMALIES",
        "GOAL_AND_MEMORY_UPDATE",
        "GENERAL_FOLLOWUP",
        "BLOCKED_OFF_TOPIC",
    ]
    is_allowed: bool = True
    guardrail_reason: Optional[str] = None
    focus_categories: List[str] = Field(default_factory=list)
    target_savings_override: Optional[float] = None
    run_spending_agent: bool = True
    run_recommendation_agent: bool = True
    run_alerting_agent: bool = True
    reasoning: str = ""
    planned_steps: List[str] = Field(default_factory=list)
    extracted_user_facts: List[str] = Field(default_factory=list)


class AgentTraceStep(BaseModel):
    """Captures autonomous reasoning, tool calls, and handoffs for UI/CLI observability."""
    step_order: int
    agent_name: str
    action_type: Literal["PLANNING", "GUARDRAIL", "TOOL_CALL", "HANDOFF", "REASONING", "MEMORY_OP", "SYNTHESIS"]
    summary: str
    tools_invoked: List[str] = Field(default_factory=list)
    input_summary: Optional[str] = None
    output_summary: Optional[str] = None
    latency_ms: float = 0.0
    llm_mode_used: str = "deterministic+llm"


class FinanceAdvisorState(TypedDict, total=False):
    """
    LangGraph state dictionary passed across all agent nodes.
    Enforces strict data dependency: Recommendation Agent consumes `spending_report`
    generated by the Spending Analysis Agent.
    """
    customer_id: str
    user_query: str
    chat_history: List[Dict[str, str]]
    memory_context: Dict[str, Any]
    execution_plan: Optional[Dict[str, Any]]
    spending_report: Optional[Dict[str, Any]]
    recommendation_report: Optional[Dict[str, Any]]
    alert_report: Optional[Dict[str, Any]]
    final_response: str
    agent_trace: List[Dict[str, Any]]

