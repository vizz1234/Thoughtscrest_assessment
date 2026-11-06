"""
Deterministic mock data generator for the Personal Finance Advisor.
Creates 3 distinct customer personas with 3 months of realistic banking transactions
(2026-08, 2026-09, 2026-10) containing embedded behavioral patterns, anomalies,
subscription creep, and budget breaches for the agents to autonomously discover.
"""

from typing import Dict, List, Tuple
from src.schemas.models import CustomerProfile, FinancialGoal, Transaction


def get_mock_profiles() -> Dict[str, CustomerProfile]:
    """Return the 3 customer personas with budgets and financial goals."""
    profiles = {
        "CUST-001": CustomerProfile(
            customer_id="CUST-001",
            name="Alex Rivera",
            persona_tag="Young Tech Professional (Food Delivery Spike & Subscription Creep)",
            monthly_net_income=6200.00,
            checking_balance=3420.50,
            savings_balance=6800.00,
            monthly_budgets={
                "Housing & Rent": 2100.00,
                "Food & Dining": 450.00,
                "Groceries": 400.00,
                "Entertainment": 250.00,
                "Subscriptions": 80.00,
                "Shopping": 350.00,
                "Transportation": 220.00,
                "Utilities": 180.00,
            },
            financial_goals=[
                FinancialGoal(
                    goal_id="GOAL-101",
                    name="House Down Payment Fund",
                    target_amount=20000.00,
                    current_amount=6800.00,
                    target_date="2027-10-01",
                    priority="High",
                ),
                FinancialGoal(
                    goal_id="GOAL-102",
                    name="Japan Vacation Fund",
                    target_amount=3500.00,
                    current_amount=1100.00,
                    target_date="2027-05-01",
                    priority="Medium",
                ),
            ],
            risk_tolerance="Moderate",
            preferred_communication_style="Data-driven, direct, action-oriented",
        ),
        "CUST-002": CustomerProfile(
            customer_id="CUST-002",
            name="Priya Sharma",
            persona_tag="Family Household Manager (Utility Spike, One-Off Appliance Anomaly & Idle Cash)",
            monthly_net_income=8500.00,
            checking_balance=14200.00,
            savings_balance=9500.00,
            monthly_budgets={
                "Housing & Rent": 2800.00,
                "Groceries": 850.00,
                "Food & Dining": 350.00,
                "Utilities": 260.00,
                "Shopping": 500.00,
                "Transportation": 380.00,
                "Entertainment": 300.00,
                "Subscriptions": 95.00,
            },
            financial_goals=[
                FinancialGoal(
                    goal_id="GOAL-201",
                    name="6-Month Family Emergency Reserve",
                    target_amount=25000.00,
                    current_amount=9500.00,
                    target_date="2027-08-01",
                    priority="High",
                ),
                FinancialGoal(
                    goal_id="GOAL-202",
                    name="Kids College 529 Contribution",
                    target_amount=10000.00,
                    current_amount=4200.00,
                    target_date="2027-12-01",
                    priority="Medium",
                ),
            ],
            risk_tolerance="Conservative",
            preferred_communication_style="Warm, structured, family-focused",
        ),
        "CUST-003": CustomerProfile(
            customer_id="CUST-003",
            name="Marcus Vance",
            persona_tag="Freelance Designer (Income Volatility, Daily Micro-Spend Drain & Cashflow Risk)",
            monthly_net_income=4100.00,
            checking_balance=890.00,
            savings_balance=1450.00,
            monthly_budgets={
                "Housing & Rent": 1650.00,
                "Food & Dining": 300.00,
                "Groceries": 320.00,
                "Transportation": 180.00,
                "Subscriptions": 110.00,
                "Entertainment": 180.00,
                "Shopping": 200.00,
                "Utilities": 150.00,
            },
            financial_goals=[
                FinancialGoal(
                    goal_id="GOAL-301",
                    name="Pay Off High-APR Credit Card Balance",
                    target_amount=2400.00,
                    current_amount=600.00,
                    target_date="2027-04-01",
                    priority="High",
                ),
                FinancialGoal(
                    goal_id="GOAL-302",
                    name="Income Smoothing Buffer (2 Months Expenses)",
                    target_amount=6500.00,
                    current_amount=1450.00,
                    target_date="2027-11-01",
                    priority="High",
                ),
            ],
            risk_tolerance="Conservative",
            preferred_communication_style="Empathetic, practical, cash-flow protective",
        ),
    }
    return profiles


def _build_alex_transactions() -> List[Transaction]:
    """
    Build 3 months of transactions for CUST-001 (Alex Rivera).
    Planted patterns:
    - Food & Dining jumps from $395 (Aug) -> $430 (Sep) -> $742 (Oct) (+72.5% MoM, 165% of $450 budget!)
      Driven by late-night DoorDash & UberEats orders.
    - Subscriptions: 6 recurring charges totaling $136.94/mo (budget $80), including overlapping
      streaming services (StreamFlix, HuluPlus, MaxCinema) and dual music apps (Spotify, Apple Music).
    - Entertainment: Exceeds $250 budget in Oct ($335) due to concert tickets & bar tabs.
    """
    txns: List[Transaction] = []
    counter = 1

    def add(date: str, merchant: str, cat: str, subcat: str, amt: float, ttype="debit", rec=False, notes=None):
        nonlocal counter
        txns.append(
            Transaction(
                transaction_id=f"TXN-1{counter:03d}",
                customer_id="CUST-001",
                date=date,
                month=date[:7],
                merchant=merchant,
                category=cat,
                subcategory=subcat,
                amount=round(amt, 2),
                transaction_type=ttype,
                is_recurring=rec,
                notes=notes,
            )
        )
        counter += 1

    for month_idx, m in enumerate(["2026-08", "2026-09", "2026-10"]):
        # Income
        add(f"{m}-01", "TechCorp Payroll Direct Deposit", "Income", "Salary", 3100.00, "credit", True)
        add(f"{m}-15", "TechCorp Payroll Direct Deposit", "Income", "Salary", 3100.00, "credit", True)

        # Housing & Rent
        add(f"{m}-01", "Skyline Apartments", "Housing & Rent", "Rent", 2100.00, "debit", True)

        # Utilities
        add(f"{m}-05", "City Power & Light", "Utilities", "Electricity", 95.00 + month_idx * 4, "debit", True)
        add(f"{m}-08", "FiberFast Internet", "Utilities", "Internet", 79.99, "debit", True)

        # Subscriptions (6 recurring charges -> $136.94 total in Oct, redundant streaming & music)
        add(f"{m}-03", "StreamFlix UHD", "Subscriptions", "Video Streaming", 22.99 if m == "2026-10" else 19.99, "debit", True, "Price increased by $3 in Oct")
        add(f"{m}-04", "HuluPlus Ad-Free", "Subscriptions", "Video Streaming", 18.99, "debit", True)
        add(f"{m}-06", "MaxCinema", "Subscriptions", "Video Streaming", 16.99, "debit", True)
        add(f"{m}-07", "Spotify Duo", "Subscriptions", "Music Streaming", 14.99, "debit", True)
        add(f"{m}-09", "Apple Music Individual", "Subscriptions", "Music Streaming", 10.99, "debit", True, "Overlapping music subscription")
        add(f"{m}-11", "CloudGym Pro App", "Subscriptions", "Fitness App", 51.99, "debit", True)

        # Groceries (~$360-$385/mo, within $400 budget)
        add(f"{m}-04", "Trader Joe's", "Groceries", "Supermarket", 118.40)
        add(f"{m}-12", "Whole Foods Market", "Groceries", "Supermarket", 134.20)
        add(f"{m}-21", "Trader Joe's", "Groceries", "Supermarket", 112.80)

        # Transportation
        add(f"{m}-06", "Metro Transit Pass", "Transportation", "Public Transit", 95.00, "debit", True)
        add(f"{m}-14", "Uber Trip", "Transportation", "Rideshare", 38.50)
        add(f"{m}-24", "Lyft Ride", "Transportation", "Rideshare", 44.20)

    # Month-specific Food & Dining, Entertainment, Shopping
    # August Food & Dining ($395.00)
    add("2026-08-03", "DoorDash - Thai Basil", "Food & Dining", "Food Delivery", 42.50)
    add("2026-08-08", "Blue Bottle Coffee", "Food & Dining", "Coffee Shop", 14.50)
    add("2026-08-11", "Chipotle Mexican Grill", "Food & Dining", "Fast Casual", 24.00)
    add("2026-08-15", "Osteria Mozza", "Food & Dining", "Restaurant", 118.00)
    add("2026-08-19", "UberEats - Sushi Express", "Food & Dining", "Food Delivery", 54.00)
    add("2026-08-23", "Sweetgreen", "Food & Dining", "Fast Casual", 28.00)
    add("2026-08-27", "DoorDash - Burger Craft", "Food & Dining", "Food Delivery", 46.00)
    add("2026-08-29", "The Ramen Joint", "Food & Dining", "Restaurant", 68.00)

    # August Entertainment & Shopping
    add("2026-08-14", "AMC Theatres", "Entertainment", "Movies", 42.00)
    add("2026-08-22", "Topgolf Lounge", "Entertainment", "Nightlife", 145.00)
    add("2026-08-18", "Amazon Marketplace", "Shopping", "Online Retail", 165.00)
    add("2026-08-25", "Uniqlo", "Shopping", "Apparel", 110.00)

    # September Food & Dining ($430.00)
    add("2026-09-02", "DoorDash - Ramen Nagi", "Food & Dining", "Food Delivery", 48.00)
    add("2026-09-05", "Blue Bottle Coffee", "Food & Dining", "Coffee Shop", 16.50)
    add("2026-09-09", "UberEats - Taco Fiesta", "Food & Dining", "Food Delivery", 44.50)
    add("2026-09-13", "Bistro Laurent", "Food & Dining", "Restaurant", 125.00)
    add("2026-09-17", "DoorDash - Curry House", "Food & Dining", "Food Delivery", 52.00)
    add("2026-09-21", "Sweetgreen", "Food & Dining", "Fast Casual", 31.00)
    add("2026-09-25", "UberEats - Late Night Pizza", "Food & Dining", "Food Delivery", 49.00)
    add("2026-09-28", "Craft Taproom & Kitchen", "Food & Dining", "Restaurant", 64.00)

    # September Entertainment & Shopping
    add("2026-09-12", "Symphony Hall", "Entertainment", "Live Events", 110.00)
    add("2026-09-26", "Arcade Bar Social", "Entertainment", "Nightlife", 95.00)
    add("2026-09-10", "Amazon Marketplace", "Shopping", "Online Retail", 142.00)
    add("2026-09-22", "Nike Run Store", "Shopping", "Apparel", 155.00)

    # October Food & Dining ($742.00 -> Massive +72.5% MoM Spike! 11 delivery orders + dining + coffee)
    add("2026-10-02", "DoorDash - Pad Thai Palace", "Food & Dining", "Food Delivery", 52.00)
    add("2026-10-04", "UberEats - Shake Shack", "Food & Dining", "Food Delivery", 44.50)
    add("2026-10-05", "Blue Bottle Coffee", "Food & Dining", "Coffee Shop", 18.00)
    add("2026-10-07", "DoorDash - Himalayan Kitchen", "Food & Dining", "Food Delivery", 58.00)
    add("2026-10-09", "UberEats - Poke Bowl Co", "Food & Dining", "Food Delivery", 41.50)
    add("2026-10-11", "Steakhouse Prime 44", "Food & Dining", "Restaurant", 148.00)
    add("2026-10-13", "DoorDash - Midnight Wings", "Food & Dining", "Food Delivery", 49.00)
    add("2026-10-15", "Blue Bottle Coffee", "Food & Dining", "Coffee Shop", 16.50)
    add("2026-10-17", "UberEats - Sushi Omakase Box", "Food & Dining", "Food Delivery", 66.00)
    add("2026-10-19", "DoorDash - Burger & Shake", "Food & Dining", "Food Delivery", 45.50)
    add("2026-10-22", "UberEats - Burrito Express", "Food & Dining", "Food Delivery", 39.00)
    add("2026-10-25", "Rooftop Tapas Lounge", "Food & Dining", "Restaurant", 112.00)
    add("2026-10-28", "DoorDash - Artisan Pizza", "Food & Dining", "Food Delivery", 52.00)

    # October Entertainment ($335.00 -> Breaches $250 budget by 34%)
    add("2026-10-10", "Ticketmaster - Indie Arena Tour", "Entertainment", "Live Events", 215.00, "debit", False, "Concert tickets spike")
    add("2026-10-18", "Velvet Cocktail Lounge", "Entertainment", "Nightlife", 82.00)
    add("2026-10-24", "Steam Games Store", "Entertainment", "Gaming", 38.00)

    # October Shopping ($410.00 -> Breaches $350 budget)
    add("2026-10-08", "Amazon Marketplace", "Shopping", "Online Retail", 125.00)
    add("2026-10-16", "Best Buy Electronics", "Shopping", "Electronics", 285.00, "debit", False, "Wireless noise-canceling headphones")

    return txns


def _build_priya_transactions() -> List[Transaction]:
    """
    Build 3 months of transactions for CUST-002 (Priya Sharma).
    Planted patterns:
    - Utilities jumped from $235 (Aug) and $242 (Sep) to $388 (Oct) (+60.3% MoM spike).
    - Shopping has an unusual single high-value transaction ($680 at HomeDepot Appliance Center in Oct),
      pushing Shopping to $845 vs $500 budget.
    - High idle cash in checking ($14,200 earning 0.01% APY) that could be swept to High-Yield Savings.
    """
    txns: List[Transaction] = []
    counter = 1

    def add(date: str, merchant: str, cat: str, subcat: str, amt: float, ttype="debit", rec=False, notes=None):
        nonlocal counter
        txns.append(
            Transaction(
                transaction_id=f"TXN-2{counter:03d}",
                customer_id="CUST-002",
                date=date,
                month=date[:7],
                merchant=merchant,
                category=cat,
                subcategory=subcat,
                amount=round(amt, 2),
                transaction_type=ttype,
                is_recurring=rec,
                notes=notes,
            )
        )
        counter += 1

    for m in ["2026-08", "2026-09", "2026-10"]:
        add(f"{m}-01", "GlobalHealth Corp Payroll", "Income", "Salary", 4250.00, "credit", True)
        add(f"{m}-15", "GlobalHealth Corp Payroll", "Income", "Salary", 4250.00, "credit", True)
        add(f"{m}-01", "Oakridge Mortgage Servicing", "Housing & Rent", "Mortgage", 2800.00, "debit", True)

        # Groceries (~$810/mo, well-managed under $850 budget)
        add(f"{m}-03", "Costco Wholesale", "Groceries", "Wholesale Club", 310.00)
        add(f"{m}-10", "Whole Foods Market", "Groceries", "Supermarket", 185.00)
        add(f"{m}-18", "Trader Joe's", "Groceries", "Supermarket", 160.00)
        add(f"{m}-25", "Safeway Family Market", "Groceries", "Supermarket", 155.00)

        # Subscriptions ($89.97/mo)
        add(f"{m}-05", "Disney+ Family Bundle", "Subscriptions", "Video Streaming", 24.99, "debit", True)
        add(f"{m}-08", "Amazon Prime Membership", "Subscriptions", "Membership", 14.99, "debit", True)
        add(f"{m}-12", "Kumon Learning Portal", "Subscriptions", "Education", 49.99, "debit", True)

        # Transportation ($340/mo)
        add(f"{m}-07", "Shell Family Auto Fuel", "Transportation", "Fuel", 165.00)
        add(f"{m}-21", "Chevron Station", "Transportation", "Fuel", 175.00)

        # Food & Dining (~$295/mo, under $350 budget)
        add(f"{m}-09", "Olive Garden Family Dinner", "Food & Dining", "Restaurant", 145.00)
        add(f"{m}-20", "Panera Bread", "Food & Dining", "Fast Casual", 72.00)
        add(f"{m}-27", "Local Pizza Kitchen", "Food & Dining", "Restaurant", 78.00)

        # Entertainment (~$210/mo)
        add(f"{m}-14", "City Science Museum & Zoo", "Entertainment", "Family Activity", 125.00)
        add(f"{m}-26", "Regal Family Cinema", "Entertainment", "Movies", 85.00)

    # Utilities (Spike in Oct)
    add("2026-08-06", "Metro Electric & Gas", "Utilities", "Electricity & Gas", 155.00, "debit", True)
    add("2026-08-11", "Comcast Home Fiber", "Utilities", "Internet", 80.00, "debit", True)
    add("2026-09-06", "Metro Electric & Gas", "Utilities", "Electricity & Gas", 162.00, "debit", True)
    add("2026-09-11", "Comcast Home Fiber", "Utilities", "Internet", 80.00, "debit", True)
    add("2026-10-06", "Metro Electric & Gas", "Utilities", "Electricity & Gas", 308.00, "debit", True, "60% utility bill surge")
    add("2026-10-11", "Comcast Home Fiber", "Utilities", "Internet", 80.00, "debit", True)

    # Shopping (Normal in Aug/Sep, Anomaly Spike in Oct)
    add("2026-08-15", "Target Department Store", "Shopping", "Household", 210.00)
    add("2026-08-24", "Carter's Kids Apparel", "Shopping", "Apparel", 145.00)
    add("2026-09-15", "Target Department Store", "Shopping", "Household", 195.00)
    add("2026-09-23", "Amazon Family", "Shopping", "Online Retail", 180.00)
    add("2026-10-12", "Target Department Store", "Shopping", "Household", 165.00)
    add("2026-10-19", "HomeDepot Appliance Center", "Shopping", "Home Repair", 680.00, "debit", False, "Emergency dishwasher replacement")

    return txns


def _build_marcus_transactions() -> List[Transaction]:
    """
    Build 3 months of transactions for CUST-003 (Marcus Vance).
    Planted patterns:
    - Income dropped in Oct ($3,600 freelance payouts vs $4,400 in Aug/Sep), creating tight cashflow.
    - High-frequency micro-spend drain: 14 specialty coffee visits ($154) + 11 rideshare trips ($297) = $451/mo!
    - Creative software & entertainment subscription creep ($168/mo vs $110 budget).
    """
    txns: List[Transaction] = []
    counter = 1

    def add(date: str, merchant: str, cat: str, subcat: str, amt: float, ttype="debit", rec=False, notes=None):
        nonlocal counter
        txns.append(
            Transaction(
                transaction_id=f"TXN-3{counter:03d}",
                customer_id="CUST-003",
                date=date,
                month=date[:7],
                merchant=merchant,
                category=cat,
                subcategory=subcat,
                amount=round(amt, 2),
                transaction_type=ttype,
                is_recurring=rec,
                notes=notes,
            )
        )
        counter += 1

    # Variable freelance income
    add("2026-08-05", "Stripe Client Payout - Acme Brand", "Income", "Freelance", 2400.00, "credit")
    add("2026-08-20", "Upwork Design Retainer", "Income", "Freelance", 2100.00, "credit")
    add("2026-09-05", "Stripe Client Payout - Nova Labs", "Income", "Freelance", 2300.00, "credit")
    add("2026-09-21", "Upwork Design Retainer", "Income", "Freelance", 2000.00, "credit")
    add("2026-10-07", "Stripe Client Payout - Indie Studio", "Income", "Freelance", 1900.00, "credit")
    add("2026-10-22", "Upwork Design Retainer", "Income", "Freelance", 1700.00, "credit")

    for m in ["2026-08", "2026-09", "2026-10"]:
        add(f"{m}-01", "Loft Works Studio Co-Living", "Housing & Rent", "Rent", 1650.00, "debit", True)
        add(f"{m}-06", "ConEdison Utility", "Utilities", "Electricity", 88.00, "debit", True)
        add(f"{m}-10", "Spectrum Broadband", "Utilities", "Internet", 65.00, "debit", True)

        # Groceries (~$290/mo)
        add(f"{m}-04", "Trader Joe's", "Groceries", "Supermarket", 145.00)
        add(f"{m}-18", "Aldi Market", "Groceries", "Supermarket", 142.00)

        # Subscriptions ($167.95/mo -> exceeds $110 budget by 52%)
        add(f"{m}-02", "Adobe Creative Cloud All Apps", "Subscriptions", "Software", 59.99, "debit", True)
        add(f"{m}-04", "Figma Professional", "Subscriptions", "Software", 15.00, "debit", True)
        add(f"{m}-08", "Midjourney Pro Plan", "Subscriptions", "Software", 60.00, "debit", True)
        add(f"{m}-12", "StreamFlix Standard", "Subscriptions", "Video Streaming", 15.99, "debit", True)
        add(f"{m}-15", "Spotify Premium", "Subscriptions", "Music Streaming", 16.97, "debit", True)

        # Entertainment & Shopping
        add(f"{m}-14", "Brooklyn Bowl & Music", "Entertainment", "Nightlife", 125.00)
        add(f"{m}-24", "Indie Cinema House", "Entertainment", "Movies", 48.00)
        add(f"{m}-16", "B&H Photo & Design Gear", "Shopping", "Equipment", 175.00)

    # Aug & Sep moderate dining & transport
    for m in ["2026-08", "2026-09"]:
        add(f"{m}-03", "Stumptown Coffee Roasters", "Food & Dining", "Coffee Shop", 42.00)
        add(f"{m}-11", "Neighborhood Thai", "Food & Dining", "Restaurant", 95.00)
        add(f"{m}-22", "Corner Taco Stand", "Food & Dining", "Fast Casual", 85.00)
        add(f"{m}-09", "Uber Trip", "Transportation", "Rideshare", 75.00)
        add(f"{m}-23", "Lyft Ride", "Transportation", "Rideshare", 68.00)

    # October Micro-Spend Explosion (10 Coffee shop transactions + 8 Rideshare trips + Dining)
    coffee_dates = ["02", "04", "06", "09", "11", "14", "17", "20", "24", "27"]
    for d in coffee_dates:
        add(f"2026-10-{d}", "Stumptown Coffee Roasters", "Food & Dining", "Coffee Shop", 11.50)

    add("2026-10-08", "DoorDash - Ramen Box", "Food & Dining", "Food Delivery", 46.00)
    add("2026-10-15", "Williamsburg Pizza Bistro", "Food & Dining", "Restaurant", 94.00)
    add("2026-10-21", "UberEats - Korean BBQ", "Food & Dining", "Food Delivery", 58.00)
    add("2026-10-28", "Artisan Bakery & Cafe", "Food & Dining", "Fast Casual", 49.00)

    rideshare_dates = ["03", "07", "10", "13", "18", "21", "25", "29"]
    for idx, d in enumerate(rideshare_dates):
        merchant = "Uber Trip" if idx % 2 == 0 else "Lyft Ride"
        add(f"2026-10-{d}", merchant, "Transportation", "Rideshare", 34.50)

    return txns


def generate_all_mock_data() -> Tuple[Dict[str, CustomerProfile], List[Transaction]]:
    """Generate all profiles and transactions for seeding the database."""
    profiles = get_mock_profiles()
    transactions = (
        _build_alex_transactions()
        + _build_priya_transactions()
        + _build_marcus_transactions()
    )
    return profiles, transactions

