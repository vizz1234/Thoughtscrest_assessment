"""SQLite data access layer for customer profiles and transaction histories."""

import json
import sqlite3
from typing import Dict, List, Optional
from src.config import DB_PATH
from src.data.mock_generator import generate_all_mock_data
from src.schemas.models import CustomerProfile, Transaction


class BankDatabase:
    """Manages SQLite persistence for mock customer profiles and transactions."""

    def __init__(self, db_path=DB_PATH):
        self.db_path = str(db_path)
        self._ensure_initialized()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_initialized(self) -> None:
        """Create tables and seed mock data if empty."""
        with self._get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS customer_profiles (
                    customer_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    profile_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    customer_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    month TEXT NOT NULL,
                    merchant TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT NOT NULL,
                    amount REAL NOT NULL,
                    transaction_type TEXT NOT NULL,
                    is_recurring INTEGER NOT NULL,
                    notes TEXT
                )
                """
            )
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM customer_profiles")
            count = cursor.fetchone()["cnt"]
            if count == 0:
                self.seed_mock_data()

    def seed_mock_data(self) -> None:
        """Reset and seed the database with the 3 rich customer personas."""
        profiles, transactions = generate_all_mock_data()
        with self._get_conn() as conn:
            conn.execute("DELETE FROM customer_profiles")
            conn.execute("DELETE FROM transactions")

            for cid, profile in profiles.items():
                conn.execute(
                    "INSERT INTO customer_profiles (customer_id, name, profile_json) VALUES (?, ?, ?)",
                    (cid, profile.name, profile.model_dump_json()),
                )

            for txn in transactions:
                conn.execute(
                    """
                    INSERT INTO transactions (
                        transaction_id, customer_id, date, month, merchant,
                        category, subcategory, amount, transaction_type, is_recurring, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        txn.transaction_id,
                        txn.customer_id,
                        txn.date,
                        txn.month,
                        txn.merchant,
                        txn.category,
                        txn.subcategory,
                        txn.amount,
                        txn.transaction_type,
                        1 if txn.is_recurring else 0,
                        txn.notes,
                    ),
                )
            conn.commit()

    def list_profiles(self) -> List[CustomerProfile]:
        """Return all available customer profiles."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT profile_json FROM customer_profiles ORDER BY customer_id").fetchall()
            return [CustomerProfile.model_validate_json(r["profile_json"]) for r in rows]

    def get_profile(self, customer_id: str) -> CustomerProfile:
        """Fetch a single customer profile by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT profile_json FROM customer_profiles WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
            if not row:
                raise ValueError(f"Customer profile '{customer_id}' not found.")
            return CustomerProfile.model_validate_json(row["profile_json"])

    def update_profile(self, profile: CustomerProfile) -> None:
        """Persist changes (such as updated budgets or goals) to a customer profile."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE customer_profiles SET name = ?, profile_json = ? WHERE customer_id = ?",
                (profile.name, profile.model_dump_json(), profile.customer_id),
            )
            conn.commit()

    def query_transactions(
        self,
        customer_id: str,
        month: Optional[str] = None,
        category: Optional[str] = None,
        transaction_type: Optional[str] = "debit",
    ) -> List[Transaction]:
        """Query transactions with optional filters."""
        sql = "SELECT * FROM transactions WHERE customer_id = ?"
        params: list = [customer_id]

        if month:
            sql += " AND month = ?"
            params.append(month)
        if category:
            sql += " AND LOWER(category) = LOWER(?)"
            params.append(category)
        if transaction_type:
            sql += " AND transaction_type = ?"
            params.append(transaction_type)

        sql += " ORDER BY date ASC, transaction_id ASC"

        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [
                Transaction(
                    transaction_id=r["transaction_id"],
                    customer_id=r["customer_id"],
                    date=r["date"],
                    month=r["month"],
                    merchant=r["merchant"],
                    category=r["category"],
                    subcategory=r["subcategory"],
                    amount=float(r["amount"]),
                    transaction_type=r["transaction_type"],
                    is_recurring=bool(r["is_recurring"]),
                    notes=r["notes"],
                )
                for r in rows
            ]


# Singleton instance for easy access across tools and agents
db = BankDatabase()

