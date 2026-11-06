"""
Generates a high-resolution (1920x1080) prototype demonstration video (`CrestBank_Prototype_Demo.mp4`)
by executing the live LangGraph multi-agent pipeline (`run_advisor_turn` & `get_proactive_session_alerts`)
across 5 key showcase scenes, rendering each scene into crisp 1080p frames, and encoding via ffmpeg.
"""

from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.agents.orchestrator import get_proactive_session_alerts, run_advisor_turn
from src.data.db import db
from src.memory.memory_manager import memory_manager

FRAMES_DIR = BASE_DIR / "docs" / "video_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FFMPEG_BIN = "/opt/homebrew/bin/ffmpeg"
OUTPUT_VIDEO = BASE_DIR / "CrestBank_Prototype_Demo.mp4"


def render_html_to_png(html_content: str, out_png: Path) -> None:
    html_file = out_png.with_suffix(".html")
    html_file.write_text(html_content, encoding="utf-8")
    subprocess.run(
        [
            CHROME_BIN,
            "--headless",
            "--disable-gpu",
            "--window-size=1920,1080",
            "--hide-scrollbars",
            f"--screenshot={out_png}",
            f"file://{html_file}",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_ui_frame(
    step_badge: str,
    step_subtitle: str,
    active_cid: str,
    user_bubble: str,
    assistant_html: str,
    trace_lines: list[str],
) -> str:
    profile = db.get_profile(active_cid)
    proactive = get_proactive_session_alerts(active_cid)
    ltm = memory_manager.get_long_term_memory(active_cid)

    goals_html = ""
    for g in profile.financial_goals:
        pct = int((g.current_amount / g.target_amount) * 100)
        goals_html += f"""
        <div style="margin-bottom:8px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;">
            <span>{g.name}</span><span>${g.current_amount:,.0f} / ${g.target_amount:,.0f} ({pct}%)</span>
          </div>
          <div style="background:#334155;height:7px;border-radius:4px;overflow:hidden;">
            <div style="background:#38bdf8;width:{pct}%;height:100%;"></div>
          </div>
        </div>
        """

    alerts_html = ""
    for a in proactive.alerts[:3]:
        border_col = "#e11d48" if a.severity == "CRITICAL" else ("#d97706" if a.severity == "WARNING" else "#2563eb")
        bg_col = "#fff1f2" if a.severity == "CRITICAL" else ("#fffbeb" if a.severity == "WARNING" else "#eff6ff")
        txt_col = "#881337" if a.severity == "CRITICAL" else ("#78350f" if a.severity == "WARNING" else "#1e3a8a")
        alerts_html += f"""
        <div style="background:{bg_col};border-left:5px solid {border_col};color:{txt_col};padding:8px 12px;border-radius:6px;margin-bottom:6px;font-size:12.5px;">
          <strong>[{a.severity} • {a.alert_type}] {a.title}</strong> ({a.metric_highlight}) — {a.description}
        </div>
        """

    facts_html = "".join(f"<div style='margin-bottom:5px;'>• {f}</div>" for f in ltm.get("learned_facts", [])[-3:])
    trace_html = "".join(f"<div style='margin-bottom:5px;font-family:monospace;font-size:11.5px;color:#cbd5e1;'>{line}</div>" for line in trace_lines)

    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0; width: 1920px; height: 1080px; overflow: hidden;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f8fafc; color: #0f172a; display: flex; flex-direction: column;
  }}
  .top-ribbon {{
    height: 60px; background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%);
    color: white; display: flex; align-items: center; justify-content: space-between;
    padding: 0 28px; border-bottom: 2px solid #38bdf8;
  }}
  .workspace {{
    flex: 1; display: flex; height: 1020px; overflow: hidden;
  }}
  .sidebar {{
    width: 340px; background: #0f172a; color: #f8fafc; padding: 20px;
    display: flex; flex-direction: column; gap: 14px; border-right: 1px solid #cbd5e1;
  }}
  .main-chat {{
    flex: 1; padding: 20px 28px; display: flex; flex-direction: column; gap: 12px; overflow: hidden;
  }}
  .log-panel {{
    width: 470px; background: #0f172a; color: #f8fafc; padding: 18px;
    border-left: 2px solid #334155; display: flex; flex-direction: column; gap: 10px;
  }}
  table {{ width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 12.5px; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: left; }}
  th {{ background: #f1f5f9; font-weight: 700; }}
</style>
</head>
<body>
  <div class="top-ribbon">
    <div style="display:flex;align-items:center;gap:14px;">
      <span style="background:#38bdf8;color:#0f172a;font-weight:800;padding:5px 12px;border-radius:999px;font-size:13px;">{step_badge}</span>
      <span style="font-size:17px;font-weight:700;">{step_subtitle}</span>
    </div>
    <div style="font-size:13px;color:#93c5fd;">CrestBank Agentic Personal Finance Advisor • LangGraph + Gemini</div>
  </div>

  <div class="workspace">
    <!-- Left Sidebar -->
    <div class="sidebar">
      <div>
        <div style="font-size:20px;font-weight:800;">🏦 CrestBank AI</div>
        <div style="font-size:12px;color:#94a3b8;">Multi-Agent Personal Finance Advisor</div>
      </div>

      <div style="background:#1e293b;padding:10px 12px;border-radius:8px;border:1px solid #38bdf8;">
        <div style="font-size:11px;color:#94a3b8;">👤 Active Customer Persona</div>
        <div style="font-size:15px;font-weight:700;color:#38bdf8;margin-top:3px;">{profile.name} ({active_cid})</div>
        <div style="font-size:11.5px;color:#cbd5e1;margin-top:4px;">{profile.persona_tag}</div>
      </div>

      <div style="display:flex;gap:10px;">
        <div style="flex:1;background:#1e293b;padding:10px;border-radius:8px;">
          <div style="font-size:11px;color:#94a3b8;">Net Income/mo</div>
          <div style="font-size:18px;font-weight:700;">${profile.monthly_net_income:,.0f}</div>
        </div>
        <div style="flex:1;background:#1e293b;padding:10px;border-radius:8px;">
          <div style="font-size:11px;color:#94a3b8;">Checking Bal</div>
          <div style="font-size:18px;font-weight:700;">${profile.checking_balance:,.0f}</div>
        </div>
      </div>

      <div style="background:#1e293b;padding:12px;border-radius:8px;">
        <div style="font-size:12px;font-weight:700;margin-bottom:8px;color:#93c5fd;">🎯 Active Financial Goals</div>
        {goals_html}
      </div>

      <div style="background:#1e293b;padding:12px;border-radius:8px;flex:1;">
        <div style="font-size:12px;font-weight:700;margin-bottom:8px;color:#fcd34d;">🧠 Long-Term Memory Store ({active_cid})</div>
        <div style="font-size:12px;color:#e2e8f0;line-height:1.45;">{facts_html}</div>
      </div>
    </div>

    <!-- Center Streamlit Main View -->
    <div class="main-chat">
      <div>
        <div style="font-size:24px;font-weight:800;">Personal Finance Advisor — {profile.name}</div>
        <div style="font-size:12.5px;color:#475569;margin-top:2px;">🔒 <strong>Isolated Session Active for {profile.name} ({active_cid})</strong> — Chat history, Short-Term Memory &amp; Long-Term Memory strictly scoped to {active_cid}.</div>
      </div>

      <div style="background:white;border:1px solid #cbd5e1;border-radius:10px;padding:12px 16px;">
        <div style="font-weight:700;font-size:13.5px;margin-bottom:8px;color:#0f172a;">
          🔔 Proactive Alerting Agent Scan — {len(proactive.alerts)} Active Signals (Overall Risk: {proactive.overall_risk_level})
        </div>
        {alerts_html}
      </div>

      <!-- User Message Bubble -->
      <div style="align-self:flex-end;background:#2563eb;color:white;padding:11px 16px;border-radius:10px;font-size:14px;font-weight:600;max-width:85%;">
        🧑 Customer ({profile.name}): "{user_bubble}"
      </div>

      <!-- Assistant Response Bubble -->
      <div style="background:white;border:1px solid #cbd5e1;border-radius:10px;padding:16px 20px;flex:1;overflow:hidden;font-size:13.2px;line-height:1.48;">
        {assistant_html}
      </div>
    </div>

    <!-- Right Developer Trace & Audit Log Panel -->
    <div class="log-panel">
      <div style="font-size:14px;font-weight:800;color:#38bdf8;border-bottom:1px solid #334155;padding-bottom:8px;">
        📂 Background Audit Log (logs/{active_cid}_traces.log)
      </div>
      <div style="font-size:11.5px;color:#94a3b8;">
        Automatically persisted behind the scenes so the customer UI stays clean while preserving full LangGraph observability:
      </div>
      <div style="background:#020617;border:1px solid #1e293b;border-radius:8px;padding:12px;flex:1;overflow:hidden;">
        {trace_html}
      </div>
    </div>
  </div>
</body>
</html>
"""


def format_trace_lines(state: dict) -> list[str]:
    lines = [
        f"<span style='color:#38bdf8;font-weight:bold;'>Customer: {state['customer_id']} | Intent: {state['execution_plan']['user_intent']}</span>",
        "<span style='color:#64748b;'>----------------------------------------------------</span>",
    ]
    for step in state.get("agent_trace", []):
        col = "#4ade80" if step["action_type"] != "GUARDRAIL" else "#f43f5e"
        lines.append(
            f"<span style='color:{col};font-weight:bold;'>▸ Step {step['step_order']}: {step['agent_name']} [{step['action_type']}]</span> ({step['latency_ms']} ms)"
        )
        lines.append(f"&nbsp;&nbsp;Tools: <code>{', '.join(step['tools_invoked'])}</code>")
        lines.append(f"&nbsp;&nbsp;Summary: {step['summary']}")
        lines.append(f"&nbsp;&nbsp;Handoff: <span style='color:#93c5fd;'>{step['output_summary']}</span><br/>")
    return lines


def main():
    db.seed_mock_data()
    memory_manager.clear_session("CUST-001")
    memory_manager.clear_session("CUST-002")
    memory_manager.clear_session("CUST-003")

    # Execute live LangGraph pipeline for our demonstration scenes
    s1_query = "Analyze my spending patterns this month, especially Food & Dining and Subscriptions, and show me how to save more."
    s1_state = run_advisor_turn("CUST-001", s1_query)

    s2_query = "I'm saving for a wedding in December — set my dining budget to $500 and show me my updated plan."
    s2_state = run_advisor_turn("CUST-001", s2_query)

    s3_query = "Are there any unusual spending spikes or budget alerts on my account, and how can I optimize my family's idle cash?"
    s3_state = run_advisor_turn("CUST-002", s3_query)

    s4_query = "give me a pasta receipe and tell me how to treat my acne"
    s4_state = run_advisor_turn("CUST-002", s4_query)

    # Scene 1: Title & Architecture Overview (taken from our technical report diagram)
    scene1_html = """
    <!DOCTYPE html><html><body style="margin:0;width:1920px;height:1080px;background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 100%);color:white;font-family:-apple-system,sans-serif;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:60px;">
      <div style="background:#38bdf8;color:#0f172a;font-weight:800;padding:8px 20px;border-radius:999px;font-size:18px;margin-bottom:20px;">PROTOTYPE DEMONSTRATION WALKTHROUGH</div>
      <h1 style="font-size:54px;margin:0 0 16px 0;">🏦 CrestBank AI Personal Finance Advisor</h1>
      <p style="font-size:24px;color:#93c5fd;max-width:1200px;margin:0 0 40px 0;">Autonomous Multi-Agent System Built with LangGraph, Google Gemini, Deterministic Financial Tools, Dual-Tier Memory &amp; Domain Guardrails</p>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:22px;max-width:1550px;width:100%;text-align:left;">
        <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.2);padding:24px;border-radius:14px;">
          <div style="font-size:22px;font-weight:800;color:#38bdf8;margin-bottom:10px;">1. Planner Agent</div>
          <div style="font-size:16px;color:#e2e8f0;line-height:1.5;">Evaluates Dual-Mode Guardrails, consults Short/Long-Term Memory, builds the ExecutionPlan, orchestrates sub-agents, and collates the final response.</div>
        </div>
        <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.2);padding:24px;border-radius:14px;">
          <div style="font-size:22px;font-weight:800;color:#4ade80;margin-bottom:10px;">2. Spending Analysis Agent</div>
          <div style="font-size:16px;color:#e2e8f0;line-height:1.5;">Reads 3-month SQLite ledger, categorizes spending, computes MoM % deltas, and detects overlapping subscriptions &amp; micro-spend habits.</div>
        </div>
        <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.2);padding:24px;border-radius:14px;">
          <div style="font-size:22px;font-weight:800;color:#fcd34d;margin-bottom:10px;">3. Recommendation Agent</div>
          <div style="font-size:16px;color:#e2e8f0;line-height:1.5;">Strictly consumes <code>SpendingAnalysisReport</code> from Spending Analysis Agent to simulate habit cuts, budget optimization, and savings goal acceleration.</div>
        </div>
        <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.2);padding:24px;border-radius:14px;">
          <div style="font-size:22px;font-weight:800;color:#fb7185;margin-bottom:10px;">4. Alerting Agent</div>
          <div style="font-size:16px;color:#e2e8f0;line-height:1.5;">Runs proactively at session start &amp; reactively in-graph to flag budget breaches (&ge;100%), statistical Z-score outliers, and cashflow burn risks.</div>
        </div>
      </div>
    </body></html>
    """

    # Scene 2: Alex Rivera (CUST-001) Full Multi-Agent Analysis + Recommendation Handoff
    s1_assistant_html = """
    <div style="font-weight:800;font-size:15px;color:#1e3a8a;margin-bottom:6px;">1. Spending Analysis (2026-10 vs. 2026-09) — Spending Analysis Agent</div>
    <div style="margin-bottom:8px;">• <strong>Monthly Net Income:</strong> <code>$6,200.00</code> | <strong>Total Spend:</strong> <code>$4,450.03</code> (<code>+14.5% MoM</code>) | <strong>Current Savings Rate:</strong> <code>28.2%</code></div>
    <table>
      <tr><th>Category</th><th>Current Spend</th><th>Budget</th><th>Utilization</th><th>MoM Change</th><th>Top Merchants</th></tr>
      <tr><td><strong>Food &amp; Dining</strong></td><td>$742.00</td><td>$450</td><td>🔴 165%</td><td>+72.6%</td><td><code>Steakhouse Prime 44 ($148)</code>, <code>Rooftop Tapas ($112)</code></td></tr>
      <tr><td><strong>Subscriptions</strong></td><td>$136.94</td><td>$80</td><td>🔴 171%</td><td>+2.2%</td><td><code>CloudGym Pro ($52)</code>, <code>StreamFlix UHD ($23)</code></td></tr>
      <tr><td><strong>Entertainment</strong></td><td>$335.00</td><td>$250</td><td>🔴 134%</td><td>+63.4%</td><td><code>Ticketmaster Tour ($215)</code>, <code>Velvet Lounge ($82)</code></td></tr>
      <tr><td><strong>Shopping</strong></td><td>$410.00</td><td>$350</td><td>🔴 117%</td><td>+38.0%</td><td><code>Best Buy ($285)</code>, <code>Amazon ($125)</code></td></tr>
    </table>
    <div style="font-weight:800;font-size:15px;color:#166534;margin:10px 0 6px 0;">2. Personalized Savings Plan — Recommendation Agent (Consumed SpendingAnalysisReport)</div>
    <div style="margin-bottom:6px;">• <strong>Total Unlockable Savings:</strong> <strong><code>$430.71/month</code></strong> (<strong><code>$5,168.52/year</code></strong>) | <strong>Savings Rate Boost:</strong> <code>28.2%</code> &rarr; <strong><code>35.2%</code></strong></div>
    <div>1. <strong>Right-Size Frequent Food Delivery Habit (11x &rarr; 5x/mo):</strong> Save <strong>$274.75/mo ($3,297/yr)</strong> across DoorDash/UberEats toward House Down Payment.</div>
    <div>2. <strong>Consolidate Overlapping Subscriptions:</strong> Cancel redundant HuluPlus, MaxCinema, Spotify Duo &amp; Apple Music to save <strong>$61.96/mo ($744/yr)</strong>.</div>
    <div>3. <strong>Normalize Entertainment &amp; Shopping Spikes:</strong> Cap at monthly budget baselines to reclaim <strong>$94.00/mo ($1,128/yr)</strong>.</div>
    """
    scene2_html = build_ui_frame(
        "SCENE 1 OF 4",
        "Full Multi-Agent Orchestration: Planner → Spending Analysis → Recommendation → Alerting (Alex Rivera, CUST-001)",
        "CUST-001",
        s1_query,
        s1_assistant_html,
        format_trace_lines(s1_state),
    )

    # Scene 3: Memory Update & Dynamic Budget Modification
    s2_assistant_html = """
    <div style="background:#eff6ff;border-left:4px solid #2563eb;padding:8px 12px;margin-bottom:10px;font-weight:600;color:#1e3a8a;">
      🧠 Memory Updated: Saved to Long-Term Profile — "Updated Food &amp; Dining monthly budget from $450 to $500" &amp; "Saving for a wedding in December."
    </div>
    <div style="font-weight:800;font-size:15px;color:#1e3a8a;margin-bottom:6px;">Updated Spending &amp; Budget Utilization (with New $500 Food &amp; Dining Cap)</div>
    <table>
      <tr><th>Category</th><th>Current Spend</th><th>Updated Budget</th><th>New Utilization</th><th>MoM Change</th><th>Status</th></tr>
      <tr><td><strong>Food &amp; Dining</strong></td><td>$742.00</td><td><strong>$500 (Updated)</strong></td><td>🔴 148% (was 165%)</td><td>+72.6%</td><td>Over new budget by $242.00</td></tr>
      <tr><td><strong>Subscriptions</strong></td><td>$136.94</td><td>$80</td><td>🔴 171%</td><td>+2.2%</td><td>6 active recurring charges</td></tr>
    </table>
    <div style="font-weight:800;font-size:15px;color:#166534;margin:10px 0 6px 0;">Updated Recommendation Plan (Incorporating Long-Term Memory)</div>
    <div>• <strong>Sidebar Long-Term Memory Store Updated:</strong> Notice the new wedding goal and $500 dining budget persisted in the left sidebar.</div>
    <div>• <strong>CloudGym Pro Protected:</strong> Recommendation Agent respected your LTM preference to keep CloudGym Pro active while targeting food delivery ($274.75/mo) and duplicate streaming apps ($61.96/mo).</div>
    """
    scene3_html = build_ui_frame(
        "SCENE 2 OF 4",
        "Dual-Tier Memory Persistence: Dynamic Budget Update & Life-Event Extraction (CUST-001)",
        "CUST-001",
        s2_query,
        s2_assistant_html,
        format_trace_lines(s2_state),
    )

    # Scene 4: Customer Isolation + Priya Sharma (CUST-002) Anomaly Detection & Idle Cash Sweep
    s3_assistant_html = """
    <div style="background:#f0fdf4;border-left:4px solid #16a34a;padding:8px 12px;margin-bottom:10px;font-weight:600;color:#166534;">
      🔒 Strict Customer Session Isolation Verified: Switched to Priya Sharma (CUST-002) — Zero chat history or memory leakage from CUST-001.
    </div>
    <div style="font-weight:800;font-size:15px;color:#9f1239;margin-bottom:6px;">1. Proactive Risk &amp; Statistical Anomaly Detection — Alerting Agent</div>
    <div style="margin-bottom:5px;">• 🚨 <strong>CRITICAL (BUDGET_BREACH):</strong> Shopping reached <strong>$845.00</strong> vs $500 budget (<code>169% used</code>, <code>+125.3% MoM</code>).</div>
    <div style="margin-bottom:5px;">• 🚨 <strong>CRITICAL (SPENDING_SPIKE — Z-Score Outlier):</strong> Detected a <strong>$680.00</strong> charge at <code>HomeDepot Appliance Center</code> (<strong>2.5x</strong> category average, <code>Z-score = 2.2</code>).</div>
    <div style="margin-bottom:5px;">• 🚨 <strong>CRITICAL (BUDGET_BREACH):</strong> Utilities surged <code>+60.3% MoM</code> to <strong>$388.00</strong> vs $260 budget (<code>149% used</code>).</div>
    <div style="margin-bottom:10px;">• ℹ️ <strong>INFO (GOAL_RISK — Idle Cash):</strong> <strong>$14,200.00</strong> sitting in checking at 0.01% APY.</div>
    <div style="font-weight:800;font-size:15px;color:#166534;margin-bottom:6px;">2. Personalized Cash &amp; Budget Optimization — Recommendation Agent</div>
    <div>• <strong>Sweep $7,400 Idle Checking Cash to 4.50% APY Vault:</strong> Generates <strong>+$333/year ($27.75/mo)</strong> risk-free interest toward your 6-Month Family Emergency Reserve.</div>
    <div>• <strong>Normalize Utilities &amp; Amortize One-Off Appliance Spike:</strong> Reclaims <strong>$473.00/mo</strong> in monthly cash flow.</div>
    """
    scene4_html = build_ui_frame(
        "SCENE 3 OF 4",
        "Customer Isolation + Z-Score Anomaly Detection & Idle Cash Sweep (Priya Sharma, CUST-002)",
        "CUST-002",
        s3_query,
        s3_assistant_html,
        format_trace_lines(s3_state),
    )

    # Scene 5: Domain Guardrail Intercepting Off-Topic Recipes & Medical Advice
    s4_assistant_html = """
    <div style="background:#fff1f2;border:2px solid #e11d48;border-radius:10px;padding:18px 22px;color:#881337;">
      <div style="font-size:18px;font-weight:800;margin-bottom:10px;">🛡️ Outside Financial Advisory Scope — Guardrail Triggered</div>
      <p style="margin:0 0 10px 0;">Hi <strong>Priya</strong>, as your <strong>CrestBank AI Personal Finance Advisor</strong>, I am strictly designed to assist with your <strong>bank accounts, transaction analysis, category budgets, overspending alerts, and savings goals</strong>.</p>
      <div style="background:white;padding:10px 14px;border-radius:6px;border:1px solid #fda4af;margin-bottom:12px;">
        <strong>Guardrail Decision:</strong> Request Declined (<code>Blocked off-topic request under category 'Cooking &amp; Recipes' / 'Medical &amp; Health Advice'</code>)<br/>
        <strong>LangGraph Short-Circuit:</strong> Routed directly <code>planner_plan_node &rarr; planner_synthesize_node</code> (0 financial tools or sub-agents invoked).
      </div>
      <div style="font-weight:700;margin-bottom:4px;">What You Can Ask Me Instead:</div>
      <div>1. <em>"Analyze my spending patterns this month and show me how to save more."</em></div>
      <div>2. <em>"Why did my Utilities or Shopping jump this month?"</em></div>
      <div>3. <em>"Are there any unusual spending spikes or budget breaches on my account?"</em></div>
    </div>
    """
    scene5_html = build_ui_frame(
        "SCENE 4 OF 4",
        "Dual-Mode Domain Guardrail: Blocking Off-Topic Queries ('pasta receipe' / 'treat my acne')",
        "CUST-002",
        s4_query,
        s4_assistant_html,
        format_trace_lines(s4_state),
    )

    scenes = [
        ("scene1.png", scene1_html, 5),
        ("scene2.png", scene2_html, 8),
        ("scene3.png", scene3_html, 7),
        ("scene4.png", scene4_html, 7),
        ("scene5.png", scene5_html, 7),
    ]

    concat_lines = []
    for fname, html_str, duration in scenes:
        out_png = FRAMES_DIR / fname
        render_html_to_png(html_str, out_png)
        concat_lines.append(f"file '{out_png}'")
        concat_lines.append(f"duration {duration}")
    # Repeat last frame for ffmpeg concat demuxer
    concat_lines.append(f"file '{FRAMES_DIR / scenes[-1][0]}'")

    concat_file = FRAMES_DIR / "slides.txt"
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")

    subprocess.run(
        [
            FFMPEG_BIN,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-vf",
            "fps=30,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            str(OUTPUT_VIDEO),
        ],
        check=True,
    )
    print(f"Video successfully created at: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()

