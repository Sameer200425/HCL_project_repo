"""
Fraud Trend Analytics Engine — Banking Intelligence Layer.
============================================================
Transforms raw fraud predictions into actionable financial intelligence.

CAPABILITIES:
  1. Temporal Pattern Analysis  — detect fraud spikes, seasonal trends
  2. Fraud Velocity Tracking    — rate-of-change anomaly detection
  3. Branch-Level Hotspot Map   — geographic risk aggregation
  4. Fraud Class Drift Detector — track which forgery type is rising
  5. Weekly / Monthly Reports   — automated summary statistics
  6. Z-Score Anomaly Flagging   — flag days with unusual fraud volume

WHY UNIQUE FOR HCL/BANKING:
  Most fraud detection stops at "fraud/genuine". This engine provides
  the INTELLIGENCE LAYER that tells risk analysts WHERE fraud is going,
  WHEN it spikes, and WHAT forgery technique is trending — enabling
  proactive branch-level interventions rather than reactive case-by-case.

USAGE:
  engine = FraudTrendEngine()
  engine.ingest(predictions_df)
  report = engine.generate_weekly_report()
  hotspots = engine.get_branch_hotspots(top_n=5)
============================================================
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ------------------------------------------------------------------ #
#  Data Models
# ------------------------------------------------------------------ #

@dataclass
class DailyFraudStat:
    date: str
    total_documents: int
    fraud_count: int
    genuine_count: int
    tampered_count: int
    forged_count: int
    fraud_rate: float            # fraud / total
    avg_confidence: float
    anomaly_score: float = 0.0  # Z-score vs rolling baseline
    is_anomaly: bool = False


@dataclass
class BranchRiskProfile:
    branch_id: str
    branch_name: str
    total_docs: int
    fraud_count: int
    fraud_rate: float
    risk_level: str             # LOW / MEDIUM / HIGH / CRITICAL
    dominant_fraud_type: str
    trend: str                  # IMPROVING / STABLE / WORSENING


@dataclass
class FraudTrendReport:
    report_date: str
    period: str
    total_documents: int
    total_fraud: int
    overall_fraud_rate: float
    fraud_rate_change_pct: float   # vs previous period
    peak_fraud_day: str
    dominant_class: str
    anomaly_days: List[str]
    branch_hotspots: List[BranchRiskProfile]
    class_breakdown: Dict[str, int]
    trend_direction: str            # INCREASING / STABLE / DECREASING
    risk_alert: Optional[str]       # None or alert message
    recommendations: List[str]


# ------------------------------------------------------------------ #
#  Core Engine
# ------------------------------------------------------------------ #

class FraudTrendEngine:
    """
    Analyzes time-series of fraud prediction records and generates
    banking intelligence reports.
    """

    FRAUD_CLASSES = {"fraud", "tampered", "forged"}
    GENUINE_CLASS = "genuine"

    RISK_THRESHOLDS = {
        "LOW":      0.05,
        "MEDIUM":   0.15,
        "HIGH":     0.30,
    }

    def __init__(
        self,
        window_days: int = 7,
        anomaly_z_threshold: float = 2.5,
    ):
        """
        Args:
            window_days          : Rolling window for baseline computation.
            anomaly_z_threshold  : Z-score threshold to flag anomaly days.
        """
        self.window_days         = window_days
        self.anomaly_z_threshold = anomaly_z_threshold
        self._records: List[Dict] = []   # raw ingested predictions

    # ---------------------------------------------------------------- #
    #  Data Ingestion
    # ---------------------------------------------------------------- #

    def ingest_prediction(
        self,
        predicted_class: str,
        confidence: float,
        timestamp: Optional[datetime] = None,
        branch_id: str = "unknown",
        branch_name: str = "Unknown Branch",
    ) -> None:
        """
        Add a single prediction record to the engine.

        Args:
            predicted_class : "genuine" | "fraud" | "tampered" | "forged"
            confidence      : Model confidence score [0, 1]
            timestamp       : When the document was processed (default = now)
            branch_id       : Bank branch identifier
            branch_name     : Human-readable branch name
        """
        self._records.append({
            "date": (timestamp or datetime.utcnow()).strftime("%Y-%m-%d"),
            "datetime": (timestamp or datetime.utcnow()).isoformat(),
            "predicted_class": predicted_class.lower(),
            "confidence": confidence,
            "branch_id": branch_id,
            "branch_name": branch_name,
            "is_fraud": predicted_class.lower() in self.FRAUD_CLASSES,
        })

    def ingest_batch(self, records: List[Dict]) -> None:
        """
        Bulk ingest a list of prediction dicts.
        Each dict must have: predicted_class, confidence, timestamp (optional),
        branch_id (optional), branch_name (optional).
        """
        for r in records:
            self.ingest_prediction(
                predicted_class=r["predicted_class"],
                confidence=r.get("confidence", 0.5),
                timestamp=r.get("timestamp"),
                branch_id=r.get("branch_id", "unknown"),
                branch_name=r.get("branch_name", "Unknown Branch"),
            )

    # ---------------------------------------------------------------- #
    #  Daily Aggregation
    # ---------------------------------------------------------------- #

    def _aggregate_by_date(self) -> List[DailyFraudStat]:
        """Group records by date and compute daily stats."""
        by_date: Dict[str, List[Dict]] = defaultdict(list)
        for r in self._records:
            by_date[r["date"]].append(r)

        stats: List[DailyFraudStat] = []
        for date in sorted(by_date.keys()):
            recs = by_date[date]
            total = len(recs)
            counts = {c: 0 for c in ["genuine", "fraud", "tampered", "forged"]}
            total_conf = 0.0
            for r in recs:
                cls = r["predicted_class"]
                if cls in counts:
                    counts[cls] += 1
                total_conf += r["confidence"]

            fraud_total = counts["fraud"] + counts["tampered"] + counts["forged"]
            stats.append(DailyFraudStat(
                date=date,
                total_documents=total,
                fraud_count=fraud_total,
                genuine_count=counts["genuine"],
                tampered_count=counts["tampered"],
                forged_count=counts["forged"],
                fraud_rate=fraud_total / max(total, 1),
                avg_confidence=total_conf / max(total, 1),
            ))

        return stats

    def _compute_anomaly_scores(
        self, daily_stats: List[DailyFraudStat]
    ) -> List[DailyFraudStat]:
        """
        Compute Z-score for each day's fraud rate against
        a rolling mean/std window.
        """
        rates = np.array([d.fraud_rate for d in daily_stats], dtype=float)
        n = len(rates)

        for i, day in enumerate(daily_stats):
            window_start = max(0, i - self.window_days)
            window = rates[window_start:i] if i > 0 else np.array([rates[i]])

            mu  = window.mean()
            std = window.std() if len(window) > 1 else 1e-6

            z = float((rates[i] - mu) / (std + 1e-8))
            day.anomaly_score = round(z, 4)
            day.is_anomaly    = abs(z) >= self.anomaly_z_threshold

        return daily_stats

    # ---------------------------------------------------------------- #
    #  Branch Hotspot Analysis
    # ---------------------------------------------------------------- #

    def get_branch_hotspots(self, top_n: int = 5) -> List[BranchRiskProfile]:
        """
        Return the branches with highest fraud risk.

        Args:
            top_n: Number of top-risk branches to return.
        """
        by_branch: Dict[str, List[Dict]] = defaultdict(list)
        for r in self._records:
            by_branch[r["branch_id"]].append(r)

        profiles: List[BranchRiskProfile] = []

        for branch_id, recs in by_branch.items():
            total = len(recs)
            fraud_recs = [r for r in recs if r["is_fraud"]]
            rate = len(fraud_recs) / max(total, 1)

            # Dominant fraud type
            class_counts: Dict[str, int] = defaultdict(int)
            for r in fraud_recs:
                class_counts[r["predicted_class"]] += 1
            dominant = max(class_counts, key=lambda k: class_counts[k]) if class_counts else "none"

            # Risk level
            t = self.RISK_THRESHOLDS
            if rate < t["LOW"]:
                risk = "LOW"
            elif rate < t["MEDIUM"]:
                risk = "MEDIUM"
            elif rate < t["HIGH"]:
                risk = "HIGH"
            else:
                risk = "CRITICAL"

            # Trend: compare first vs last half of records
            half = max(total // 2, 1)
            early_rate = sum(1 for r in recs[:half] if r["is_fraud"]) / half
            late_rate  = sum(1 for r in recs[half:] if r["is_fraud"]) / max(total - half, 1)

            if late_rate > early_rate * 1.2:
                trend = "WORSENING"
            elif late_rate < early_rate * 0.8:
                trend = "IMPROVING"
            else:
                trend = "STABLE"

            profiles.append(BranchRiskProfile(
                branch_id=branch_id,
                branch_name=recs[0].get("branch_name", branch_id),
                total_docs=total,
                fraud_count=len(fraud_recs),
                fraud_rate=round(rate, 4),
                risk_level=risk,
                dominant_fraud_type=dominant,
                trend=trend,
            ))

        profiles.sort(key=lambda p: p.fraud_rate, reverse=True)
        return profiles[:top_n]

    # ---------------------------------------------------------------- #
    #  Fraud Class Drift
    # ---------------------------------------------------------------- #

    def get_class_drift(self, days: int = 30) -> Dict[str, Any]:
        """
        Detect which fraud class is increasing/decreasing over last N days.

        Returns a drift summary per class.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        recent = [r for r in self._records if r["date"] >= cutoff]
        older  = [r for r in self._records if r["date"] < cutoff]

        def class_rate(recs: List[Dict], cls: str) -> float:
            if not recs:
                return 0.0
            return sum(1 for r in recs if r["predicted_class"] == cls) / len(recs)

        drift = {}
        for cls in ["fraud", "tampered", "forged", "genuine"]:
            old_rate    = class_rate(older, cls)
            recent_rate = class_rate(recent, cls)
            change_pct  = ((recent_rate - old_rate) / max(old_rate, 1e-6)) * 100

            if change_pct > 20:
                direction = "INCREASING"
            elif change_pct < -20:
                direction = "DECREASING"
            else:
                direction = "STABLE"

            drift[cls] = {
                "old_rate":    round(old_rate, 4),
                "recent_rate": round(recent_rate, 4),
                "change_pct":  round(change_pct, 2),
                "direction":   direction,
            }

        return drift

    # ---------------------------------------------------------------- #
    #  Report Generation
    # ---------------------------------------------------------------- #

    def generate_weekly_report(
        self, end_date: Optional[datetime] = None
    ) -> FraudTrendReport:
        """
        Generate a 7-day fraud intelligence report.
        """
        return self._generate_period_report(period_days=7, label="Weekly", end_date=end_date)

    def generate_monthly_report(
        self, end_date: Optional[datetime] = None
    ) -> FraudTrendReport:
        """
        Generate a 30-day fraud intelligence report.
        """
        return self._generate_period_report(period_days=30, label="Monthly", end_date=end_date)

    def _generate_period_report(
        self,
        period_days: int,
        label: str,
        end_date: Optional[datetime] = None,
    ) -> FraudTrendReport:
        end   = end_date or datetime.utcnow()
        start = end - timedelta(days=period_days)
        prev_start = start - timedelta(days=period_days)

        end_str   = end.strftime("%Y-%m-%d")
        start_str = start.strftime("%Y-%m-%d")
        prev_str  = prev_start.strftime("%Y-%m-%d")

        current = [r for r in self._records if start_str <= r["date"] <= end_str]
        previous = [r for r in self._records if prev_str <= r["date"] < start_str]

        total    = len(current)
        fraud_c  = sum(1 for r in current if r["is_fraud"])
        fraud_rate = fraud_c / max(total, 1)

        prev_total = len(previous)
        prev_fraud = sum(1 for r in previous if r["is_fraud"])
        prev_rate  = prev_fraud / max(prev_total, 1)

        rate_change_pct = ((fraud_rate - prev_rate) / max(prev_rate, 1e-6)) * 100

        # Daily stats for anomaly detection
        daily_raw = self._aggregate_by_date()
        current_daily = [d for d in daily_raw if start_str <= d.date <= end_str]
        current_daily = self._compute_anomaly_scores(current_daily)
        anomaly_days = [d.date for d in current_daily if d.is_anomaly]

        # Peak fraud day
        peak = max(current_daily, key=lambda d: d.fraud_rate, default=None)
        peak_day = peak.date if peak else "N/A"

        # Class breakdown
        class_bd: Dict[str, int] = defaultdict(int)
        for r in current:
            class_bd[r["predicted_class"]] += 1

        # Dominant fraud class
        fraud_class_counts = {k: v for k, v in class_bd.items() if k in self.FRAUD_CLASSES}
        dominant = max(fraud_class_counts, key=lambda k: fraud_class_counts[k]) if fraud_class_counts else "none"

        # Trend direction
        if rate_change_pct > 15:
            trend_dir = "INCREASING"
        elif rate_change_pct < -15:
            trend_dir = "DECREASING"
        else:
            trend_dir = "STABLE"

        # Risk alert
        alert = None
        if fraud_rate > 0.30:
            alert = f"CRITICAL: Fraud rate {fraud_rate:.1%} exceeds 30% threshold. Immediate branch audit required."
        elif trend_dir == "INCREASING" and rate_change_pct > 50:
            alert = f"WARNING: Fraud rate increased {rate_change_pct:.1f}% vs previous period. Escalate to Risk Committee."

        # Recommendations
        recs = self._generate_recommendations(
            fraud_rate, rate_change_pct, dominant, anomaly_days
        )

        return FraudTrendReport(
            report_date=end_str,
            period=f"{label} ({start_str} to {end_str})",
            total_documents=total,
            total_fraud=fraud_c,
            overall_fraud_rate=round(fraud_rate, 4),
            fraud_rate_change_pct=round(rate_change_pct, 2),
            peak_fraud_day=peak_day,
            dominant_class=dominant,
            anomaly_days=anomaly_days,
            branch_hotspots=self.get_branch_hotspots(top_n=3),
            class_breakdown=dict(class_bd),
            trend_direction=trend_dir,
            risk_alert=alert,
            recommendations=recs,
        )

    @staticmethod
    def _generate_recommendations(
        fraud_rate: float,
        change_pct: float,
        dominant_class: str,
        anomaly_days: List[str],
    ) -> List[str]:
        """Generate contextual risk recommendations."""
        recs = []

        if fraud_rate > 0.25:
            recs.append("Activate Level-2 scrutiny for all incoming documents.")

        if change_pct > 30:
            recs.append(
                "Fraud rate rising sharply — notify branch managers for targeted training."
            )

        if dominant_class == "forged":
            recs.append(
                "Forged documents are the primary threat. "
                "Deploy UV lamp verification at teller stations."
            )
        elif dominant_class == "tampered":
            recs.append(
                "Tampering is the dominant fraud type. "
                "Enable digital watermark verification on high-value instruments."
            )
        elif dominant_class == "fraud":
            recs.append(
                "Direct fraud submissions detected. "
                "Cross-reference with CIBIL/NSDL databases."
            )

        if len(anomaly_days) > 2:
            recs.append(
                f"{len(anomaly_days)} anomalous days detected — "
                "review CCTV footage and staff logs for those dates."
            )

        if not recs:
            recs.append("Fraud levels within normal parameters. Continue standard monitoring.")

        return recs

    def to_json(self, report: FraudTrendReport) -> str:
        """Serialize report to JSON."""
        return json.dumps(asdict(report), indent=2, default=str)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Quick summary of all ingested data."""
        total = len(self._records)
        fraud = sum(1 for r in self._records if r["is_fraud"])
        classes: Dict[str, int] = defaultdict(int)
        for r in self._records:
            classes[r["predicted_class"]] += 1

        return {
            "total_records": total,
            "total_fraud": fraud,
            "overall_fraud_rate": round(fraud / max(total, 1), 4),
            "class_distribution": dict(classes),
            "date_range": {
                "earliest": min((r["date"] for r in self._records), default="N/A"),
                "latest":   max((r["date"] for r in self._records), default="N/A"),
            },
            "unique_branches": len({r["branch_id"] for r in self._records}),
        }


# ------------------------------------------------------------------ #
#  CLI Demo
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import random
    from datetime import datetime, timedelta

    print("=" * 60)
    print("Fraud Trend Analytics Engine — Demo")
    print("=" * 60)

    engine = FraudTrendEngine(window_days=7)

    branches = [
        ("BR001", "Mumbai Central"),
        ("BR002", "Delhi Connaught Place"),
        ("BR003", "Chennai Anna Nagar"),
        ("BR004", "Bangalore Koramangala"),
    ]
    classes = ["genuine"] * 7 + ["fraud", "tampered", "forged"]

    base_date = datetime.utcnow() - timedelta(days=30)
    for i in range(300):
        ts = base_date + timedelta(hours=random.randint(0, 720))
        bid, bname = random.choice(branches)
        engine.ingest_prediction(
            predicted_class=random.choice(classes),
            confidence=random.uniform(0.6, 0.99),
            timestamp=ts,
            branch_id=bid,
            branch_name=bname,
        )

    report = engine.generate_monthly_report()
    print(f"\nPeriod          : {report.period}")
    print(f"Total Documents : {report.total_documents}")
    print(f"Total Fraud     : {report.total_fraud}")
    print(f"Fraud Rate      : {report.overall_fraud_rate:.2%}")
    print(f"Rate Change     : {report.fraud_rate_change_pct:+.1f}% vs previous")
    print(f"Trend Direction : {report.trend_direction}")
    print(f"Dominant Class  : {report.dominant_class}")
    print(f"Anomaly Days    : {len(report.anomaly_days)}")
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  • {rec}")

    print("\nBranch Hotspots:")
    for b in report.branch_hotspots:
        print(f"  [{b.risk_level:8s}] {b.branch_name:<28} Fraud: {b.fraud_rate:.2%}  Trend: {b.trend}")
