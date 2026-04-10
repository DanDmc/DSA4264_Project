"""
Aggregate Validation Results
------------------------------
Loads results from all four embedding variants and produces a single
summary JSON for easy comparison. This is the file you open first
to decide which approach to use on the full dataset.

Output structure:
- comparison_table: side-by-side aggregate metrics (the key decision table)
- per_category: per-category breakdown for each variant
- recommendation: brief interpretation of results
- variant_configs: what each variant actually did

Run after all four validate_v*.py scripts have produced their JSON outputs.
"""

import json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
VARIANT_FILES = {
    "v1_baseline": SCRIPT_DIR / "results_v1_baseline.json",
    "v2_denoised": SCRIPT_DIR / "results_v2_denoised.json",
    "v3_sentence_avg": SCRIPT_DIR / "results_v3_sentence_avg.json",
    "v4_sentence_weighted": SCRIPT_DIR / "results_v4_sentence_weighted.json",
}

OUTPUT_FILE = SCRIPT_DIR / "validation_summary.json"


def load_variant(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_table(variants):
    """
    Side-by-side table of the metrics that matter for choosing an approach.
    Designed to be copy-pasteable into a report or read at a glance.
    """
    table = {}
    for name, data in variants.items():
        retrieval = data["aggregate"]["retrieval"]
        separation = data["aggregate"]["separation"]
        table[name] = {
            "MRR": retrieval["MRR"],
            "Top1": retrieval["Top1"],
            "Top2": retrieval["Top2"],
            "separation_gap": separation["separation_gap"],
            "separation_ratio": separation["separation_ratio"],
            "misalignment_rate": separation["misalignment_rate"],
        }
    return table


def build_per_category_comparison(variants):
    """Per-category MRR and separation gap across all variants."""
    # collect all categories from first variant
    first = next(iter(variants.values()))
    categories = sorted(first["per_category"].keys())

    per_cat = {}
    for cat in categories:
        per_cat[cat] = {}
        for name, data in variants.items():
            cat_data = data["per_category"].get(cat, {})
            per_cat[cat][name] = {
                "MRR": cat_data.get("retrieval", {}).get("MRR"),
                "separation_gap": cat_data.get("separation", {}).get("separation_gap"),
                "misalignment_rate": cat_data.get("separation", {}).get("misalignment_rate"),
            }
    return per_cat


def pick_recommendation(table):
    """
    Auto-generate a short interpretation. Not meant to replace human
    judgment — just a starting point for the report write-up.
    """
    # rank by MRR first, then separation_gap as tiebreaker
    ranked = sorted(
        table.items(),
        key=lambda x: (x[1]["MRR"], x[1]["separation_gap"]),
        reverse=True,
    )

    best_name, best_metrics = ranked[0]

    lines = [
        f"Best performer: {best_name}",
        f"  MRR={best_metrics['MRR']}, separation_gap={best_metrics['separation_gap']}, "
        f"misalignment_rate={best_metrics['misalignment_rate']}",
        "",
        "Ranking by MRR (then separation gap as tiebreaker):",
    ]

    for i, (name, m) in enumerate(ranked, 1):
        lines.append(
            f"  {i}. {name}: MRR={m['MRR']}, gap={m['separation_gap']}, "
            f"misalign={m['misalignment_rate']}"
        )

    return "\n".join(lines)


def main():
    print("Aggregating validation results")
    print("=" * 50)

    # load all variants
    variants = {}
    missing = []
    for name, path in VARIANT_FILES.items():
        if path.exists():
            variants[name] = load_variant(path)
            print(f"  Loaded: {name}")
        else:
            missing.append(name)
            print(f"  MISSING: {name} — skipping")

    if not variants:
        print("No variant results found. Run validate_v*.py scripts first.")
        return

    if missing:
        print(f"\nWarning: {len(missing)} variant(s) missing: {missing}")
        print("Summary will be partial.\n")

    comparison = build_comparison_table(variants)
    per_category = build_per_category_comparison(variants)
    recommendation = pick_recommendation(comparison)

    # collect configs
    configs = {name: data.get("config", {}) for name, data in variants.items()}

    summary = {
        "generated_at": datetime.now().isoformat(),
        "variants_included": list(variants.keys()),
        "comparison_table": comparison,
        "per_category": per_category,
        "recommendation": recommendation,
        "variant_configs": configs,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"\n{'─' * 50}")
    print("COMPARISON TABLE:")
    print(f"{'─' * 50}")

    # pretty-print the comparison
    header = f"{'Variant':<25} {'MRR':>6} {'Top1':>6} {'Top2':>6} {'Gap':>8} {'Ratio':>7} {'Misalign':>9}"
    print(header)
    print("-" * len(header))
    for name, m in comparison.items():
        print(
            f"{name:<25} {m['MRR']:>6.4f} {m['Top1']:>6.4f} {m['Top2']:>6.4f} "
            f"{m['separation_gap']:>8.4f} {m['separation_ratio']:>7.4f} {m['misalignment_rate']:>9.4f}"
        )

    print(f"\n{recommendation}")


if __name__ == "__main__":
    main()
