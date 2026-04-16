"""
Derive per-skill dominant knowledge type using Wilson confidence intervals.

Goal:
    Avoid arbitrary thresholds (e.g., 0.7) when deciding whether a skill is
    mostly "applied" or mostly "theoretical".

Method:
    For each skill_label, estimate applied share p = n_applied / n_total and
    compute a Wilson confidence interval for p.

Decision rule:
    - applied     if CI lower bound > 0.5
    - theoretical if CI upper bound < 0.5
    - mixed       otherwise

Usage:
    python -m src.analysis.skill_metrics.skill_knowledge_dominance
    python -m src.analysis.skill_metrics.skill_knowledge_dominance --confidence 0.99
    python -m src.analysis.skill_metrics.skill_knowledge_dominance --output results/skill_knowledge_dominance.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from src.config import MODULE_SKILL_PAIRS, RESULTS_DIR


def _read_pairs_csv(path: Path) -> pd.DataFrame:
    """Read module_skill_pairs.csv, handling accidental extra first line."""
    df = pd.read_csv(path)
    required = {"module_code", "skill_label", "knowledge_type"}
    if required.issubset(set(df.columns)):
        return df

    # Some generated files contain an extra first row ('module_skill_pairs')
    # before the real header row.
    df = pd.read_csv(path, skiprows=1)
    if required.issubset(set(df.columns)):
        return df

    raise ValueError(
        f"Could not find required columns {sorted(required)} in {path}. "
        f"Found: {list(df.columns)}"
    )


def _wilson_interval(successes: int, total: int, confidence: float) -> tuple[float, float]:
    """Wilson score interval for Bernoulli proportion."""
    if total <= 0:
        return (0.0, 1.0)

    alpha = 1.0 - confidence
    z = NormalDist().inv_cdf(1.0 - alpha / 2.0)

    p_hat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total

    center = (p_hat + z2 / (2.0 * total)) / denom
    half = (z / denom) * ((p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total)) ** 0.5)
    return (max(0.0, center - half), min(1.0, center + half))


def build_dominance_table(df: pd.DataFrame, confidence: float) -> pd.DataFrame:
    """Build per-skill dominance summary with Wilson CI."""
    x = df.copy()
    x["skill_label"] = x["skill_label"].astype(str).str.strip()
    x["skill_label_norm"] = x["skill_label"].str.lower().str.replace(r"\s+", " ", regex=True)
    x["knowledge_type_norm"] = x["knowledge_type"].astype(str).str.strip().str.lower()

    # Keep only valid applied/theoretical rows for this analysis.
    x = x[
        x["skill_label_norm"].ne("")
        & x["skill_label_norm"].ne("nan")
        & x["knowledge_type_norm"].isin(["applied", "theoretical"])
    ]

    # Most frequent original casing per normalized skill label.
    canonical = (
        x.groupby(["skill_label_norm", "skill_label"])
        .size()
        .reset_index(name="freq")
        .sort_values(["skill_label_norm", "freq"], ascending=[True, False])
        .drop_duplicates("skill_label_norm")
        .set_index("skill_label_norm")["skill_label"]
    )

    grouped = (
        x.groupby("skill_label_norm")["knowledge_type_norm"]
        .value_counts()
        .unstack(fill_value=0)
        .rename(columns={"applied": "n_applied", "theoretical": "n_theoretical"})
    )

    if "n_applied" not in grouped.columns:
        grouped["n_applied"] = 0
    if "n_theoretical" not in grouped.columns:
        grouped["n_theoretical"] = 0

    grouped["n_total"] = grouped["n_applied"] + grouped["n_theoretical"]
    grouped["applied_share"] = grouped["n_applied"] / grouped["n_total"]

    ci = grouped.apply(
        lambda r: _wilson_interval(int(r["n_applied"]), int(r["n_total"]), confidence), axis=1
    )
    grouped["applied_ci_low"] = ci.map(lambda t: t[0])
    grouped["applied_ci_high"] = ci.map(lambda t: t[1])

    def _label(row: pd.Series) -> str:
        if row["applied_ci_low"] > 0.5:
            return "applied"
        if row["applied_ci_high"] < 0.5:
            return "theoretical"
        return "mixed"

    grouped["dominant_knowledge_type"] = grouped.apply(_label, axis=1)
    grouped["confidence_level"] = confidence

    out = grouped.reset_index().rename(columns={"skill_label_norm": "skill_label_key"})
    out["skill_label"] = out["skill_label_key"].map(canonical).fillna(out["skill_label_key"])

    return out[
        [
            "skill_label",
            "skill_label_key",
            "n_total",
            "n_applied",
            "n_theoretical",
            "applied_share",
            "applied_ci_low",
            "applied_ci_high",
            "dominant_knowledge_type",
            "confidence_level",
        ]
    ].sort_values(["n_total", "applied_share"], ascending=[False, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive dominant knowledge type per skill using Wilson CI."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=MODULE_SKILL_PAIRS,
        help=f"Path to module_skill_pairs.csv (default: {MODULE_SKILL_PAIRS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "skill_knowledge_dominance_wilson.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for Wilson interval (default: 0.95).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.confidence < 1.0):
        raise ValueError("--confidence must be in (0, 1).")

    df = _read_pairs_csv(args.input)
    out = build_dominance_table(df, args.confidence)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    counts = out["dominant_knowledge_type"].value_counts().to_dict()
    print(f"Saved: {args.output}")
    print(f"Skills analyzed: {len(out):,}")
    print("Dominant labels:")
    print(f"  applied:     {counts.get('applied', 0):,}")
    print(f"  theoretical: {counts.get('theoretical', 0):,}")
    print(f"  mixed:       {counts.get('mixed', 0):,}")


if __name__ == "__main__":
    main()

