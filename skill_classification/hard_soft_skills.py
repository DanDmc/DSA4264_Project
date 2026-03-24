"""

Usage
-----
    python hard_soft_skills.py \
        --module-skill-pairs modules_with_skills.csv \
        --jobs-skills         jobs_processed_with_skills_relevant.csv

Outputs
-------
    data/processed/hard_soft_skill_gap_metrics.json
    data/processed/uncovered_job_skills_by_type.csv
    data/processed/course_skill_type_assignment.csv
    data/processed/modules_with_skill_types.csv

Requirements
------------
    pip install pandas inflect
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import inflect
import pandas as pd

# ---------------------------------------------------------------------------
# Seed lexicon for hard/soft classification  (FIRST PASS)
# ---------------------------------------------------------------------------
# These are unambiguous ground-truth labels.  Keep entries lowercase and
# in the same canonical form produced by canonicalize() below.
# Extend this list as your domain warrants — more entries = fewer unknowns.

HARD_SKILL_LEXICON: frozenset[str] = frozenset({
    # ── Programming languages & tools ────────────────────────────────────
    "python", "r", "java", "c++", "c", "javascript", "typescript",
    "sql", "nosql", "matlab", "stata", "spss", "sas", "excel",
    "tableau", "power bi", "git", "linux", "bash", "tensorflow",
    "pytorch", "keras", "spark", "hadoop", "aws", "azure", "gcp",
    "docker", "kubernetes", "uml", "gis", "qgis", "arcgis", "cad",
    "autocad", "solidworks", "labview", "vhdl",

    # ── Data & analytics ─────────────────────────────────────────────────
    "machine learning", "deep learning", "data analysis", "data analytics",
    "statistical analysis", "regression analysis", "hypothesis testing",
    "data visualisation", "data visualization", "natural language processing",
    "computer vision", "time series analysis", "experimental design",
    "data mining", "predictive modelling", "predictive modeling",
    "multivariate analysis", "bayesian analysis", "econometric analysis",
    "quantitative research", "qualitative research", "research methodology",
    "systematic review", "literature review", "survey design",
    "data science", "bioinformatics", "computational biology",

    # ── Software & systems engineering ───────────────────────────────────
    "software development", "software engineering", "object oriented programming",
    "agile methodology", "version control", "unit testing", "system design",
    "database management", "algorithm design", "algorithm", "algorithms",
    "data structure", "operating system", "network security", "cybersecurity",
    "cloud computing", "devops", "api design",

    # ── Finance & accounting ─────────────────────────────────────────────
    "financial modelling", "financial modeling", "financial statement analysis",
    "valuation", "portfolio management", "risk management", "derivatives pricing",
    "financial accounting", "management accounting", "auditing", "taxation",
    "corporate finance", "investment analysis",

    # ── Life sciences & medicine ──────────────────────────────────────────
    "gene sequencing", "pcr", "cell culture", "spectroscopy", "chromatography",
    "molecular biology", "immunology", "pharmacology", "clinical trial design",
    "medical imaging", "genomics", "proteomics",

    # ── Engineering & physical sciences ──────────────────────────────────
    "circuit design", "thermodynamics", "structural analysis", "fluid dynamics",
    "organic synthesis", "material characterisation", "finite element analysis",
    "control system", "signal processing", "robotic", "robotics",

    # ── Law & policy ─────────────────────────────────────────────────────
    "legal research", "contract drafting", "legal writing", "case analysis",
    "legal analysis", "statutory interpretation", "dispute resolution",
    "arbitration", "litigation",

    # ── Design & architecture ─────────────────────────────────────────────
    "3d modelling", "3d modeling", "technical drawing", "architectural drawing",
    "urban planning", "structural design", "acoustic design",

    # ── Other technical ───────────────────────────────────────────────────
    "project management", "operations management", "supply chain management",
    "accounting", "financial market", "programming", "simulation",
    "modelling", "modeling",
})

SOFT_SKILL_LEXICON: frozenset[str] = frozenset({
    # ── Communication ─────────────────────────────────────────────────────
    "communication", "written communication", "oral communication",
    "academic writing", "technical writing", "report writing", "writing",
    "presentation", "public speaking", "storytelling",
    "intercultural communication", "english", "active listening",

    # ── Thinking & reasoning ─────────────────────────────────────────────
    "critical thinking", "analytical thinking", "problem solving",
    "creative thinking", "systems thinking", "ethical reasoning",
    "decision making", "logical reasoning",

    # ── Interpersonal & professional ─────────────────────────────────────
    "teamwork", "collaboration", "leadership", "negotiation",
    "conflict resolution", "mentoring", "coaching",
    "cross cultural communication", "stakeholder management",

    # ── Self-management ───────────────────────────────────────────────────
    "time management", "self directed learning", "adaptability",
    "independent research", "research", "entrepreneurship",
    "innovation", "planning",

    # ── Academic skills ───────────────────────────────────────────────────
    "peer review", "academic integrity", "information literacy",
    "source evaluation", "argument construction",
})

# Build a combined lookup: skill -> type  (used in first-pass classification)
_LEXICON_MAP: dict[str, str] = {
    **{s: "hard" for s in HARD_SKILL_LEXICON},
    **{s: "soft" for s in SOFT_SKILL_LEXICON},
}


# ---------------------------------------------------------------------------
# Canonicalization  (mirrors postprocess_skills.py — keep in sync)
# ---------------------------------------------------------------------------
_INFLECT = inflect.engine()

_PROTECT_SINGULARIZE: frozenset[str] = frozenset({
    "ethics", "economics", "statistics", "mathematics", "physics", "politics",
    "logistics", "linguistics", "genetics", "mechanics", "dynamics",
    "electronics", "robotics", "hydraulics", "acoustics", "thermodynamics",
    "analysis", "basis", "thesis", "axis", "crisis", "series", "species",
    "news", "data", "media", "criteria", "phenomena", "corpus",
    "bioinformatics", "informatics", "forensics",
})

_SYNONYM_MAP: dict[str, str] = {
    "algorithmic":          "algorithm design",
    "algorithms":           "algorithm",
    "designing algorithm":  "algorithm design",
    "algorithms design":    "algorithm design",
    "algorithms search":    "search algorithm",
    "machine algorithm":    "machine learning",
    "machine algorithms":   "machine learning",
    "machine learn algorithm": "machine learning",
    "analytic":             "analytical thinking",
    "analytical":           "analytical thinking",
    "write":                "writing",
    "writings":             "writing",
    "present":              "presentation",
    "presentations":        "presentation",
    "communicating":        "communication",
    "communications":       "communication",
    "researcher":           "research",
    "researching":          "research",
    "decision make":        "decision making",
    "problem solve":        "problem solving",
    "problem-solving":      "problem solving",
    "data analyses":        "data analysis",
    "modeling":             "modelling",
    "financial modeling":   "financial modelling",
    "predictive modeling":  "predictive modelling",
}


def normalize_skill(skill: str) -> str:
    """Lowercase + collapse whitespace (stable key for counters/sets)."""
    return " ".join(str(skill).strip().lower().split())


def canonicalize(skill: str) -> str:
    """Canonical form: normalize -> singularize last word -> apply synonym map."""
    skill = normalize_skill(skill)
    words = skill.split()
    if not words:
        return skill
    last = words[-1]
    if last not in _PROTECT_SINGULARIZE:
        singular = _INFLECT.singular_noun(last)
        if singular:
            words[-1] = singular
    skill = " ".join(words)
    return _SYNONYM_MAP.get(skill, skill)


# ---------------------------------------------------------------------------
# Skill list parser
# ---------------------------------------------------------------------------

def parse_skill_list(value: object) -> list[str]:
    """Parse a skills cell: JSON list string or legacy comma-separated string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
    return [x.strip() for x in text.split(",") if x.strip()]


def unique_canonical_skills(skills: Iterable[str]) -> set[str]:
    return {canonicalize(s) for s in skills if canonicalize(s)}


# ---------------------------------------------------------------------------
# Job demand counters
# ---------------------------------------------------------------------------

def build_job_demand_counters(jobs_df: pd.DataFrame) -> tuple[Counter, Counter]:
    """Count how many job postings mention each canonical hard / soft skill."""
    hard_counts: Counter = Counter()
    soft_counts: Counter = Counter()
    for row in jobs_df.itertuples(index=False):
        hard_raw = parse_skill_list(getattr(row, "hard_skills", None))
        soft_raw = parse_skill_list(getattr(row, "soft_skills", None))
        # Canonicalize job skills too so they match canonicalized course skills
        hard_counts.update(unique_canonical_skills(hard_raw))
        soft_counts.update(unique_canonical_skills(soft_raw))
    return hard_counts, soft_counts


# ---------------------------------------------------------------------------
# Two-pass skill type classifier
# ---------------------------------------------------------------------------

def classify_skill(
    skill: str,
    hard_job_counts: Counter,
    soft_job_counts: Counter,
) -> str:
    """Return 'hard', 'soft', or 'unknown' for a canonical skill string.

    Pass 1 — Lexicon: if the skill is in the curated seed dictionary,
              use that label immediately.  This provides ground-truth
              anchoring and prevents discipline-specific or academic
              soft skills from being misclassified as 'unknown'.

    Pass 2 — Job demand: for skills not in the lexicon, fall back to
              comparing hard vs soft mention counts in job postings
              (the v1 approach), which works well for industry-facing
              technical skills.

    A skill is only 'unknown' if it appears in neither the lexicon nor
    the jobs vocabulary at all.
    """
    # Pass 1: lexicon lookup
    if skill in _LEXICON_MAP:
        return _LEXICON_MAP[skill]

    # Pass 2: job-demand counts
    hard_w = hard_job_counts.get(skill, 0)
    soft_w = soft_job_counts.get(skill, 0)
    if hard_w == 0 and soft_w == 0:
        return "unknown"
    return "hard" if hard_w >= soft_w else "soft"


# ---------------------------------------------------------------------------
# Module level classifier
# ---------------------------------------------------------------------------

def get_module_level(code: str) -> str:
    """Return 'foundational', 'intermediate', or 'advanced' from module code.

    NUS module codes embed the level in the first digit of the numeric part:
      1000-2000  foundational  — breadth / introductory modules
                                 Expected to have lower job-alignment;
                                 being broad is the point, not a gap.
      3000-4000  intermediate  — core disciplinary modules
                                 Should show meaningful alignment.
      5000+      advanced      — specialist / graduate modules
                                 Highest expected alignment with job demands.

    This segmentation directly addresses the limitation that job-alignment
    is not the only measure of a course's value: foundational modules are
    not penalised for teaching breadth rather than vocational skills.
    """
    m = re.search(r"(\d)", str(code))
    if not m:
        return "unknown"
    digit = int(m.group(1))
    if digit <= 2:
        return "foundational"
    if digit <= 4:
        return "intermediate"
    return "advanced"




def compute_metrics(
    module_skill_pairs_path: Path,
    jobs_skills_path: Path,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    modules_df = pd.read_csv(module_skill_pairs_path)
    jobs_df = pd.read_csv(jobs_skills_path)

    # ── Validate columns ─────────────────────────────────────────────────
    required_module_cols = {"module code", "skill"}
    missing_module = required_module_cols - set(modules_df.columns)
    if missing_module:
        raise ValueError(f"Missing module columns: {sorted(missing_module)}")

    required_job_cols = {"hard_skills", "soft_skills"}
    missing_job = required_job_cols - set(jobs_df.columns)
    if missing_job:
        raise ValueError(f"Missing job columns: {sorted(missing_job)}")

    # ── Canonicalize course skills ────────────────────────────────────────
    modules_df["skill_canon"] = modules_df["skill"].astype(str).map(canonicalize)
    modules_df = modules_df[modules_df["skill_canon"] != ""].copy()

    course_skill_counts = modules_df["skill_canon"].value_counts()
    course_skill_set = set(course_skill_counts.index)

    # ── Build job demand counters (with canonicalization on job side) ─────
    hard_job_counts, soft_job_counts = build_job_demand_counters(jobs_df)
    hard_demand_total = sum(hard_job_counts.values())
    soft_demand_total = sum(soft_job_counts.values())
    total_job_demand = hard_demand_total + soft_demand_total

    # ── Demand-Weighted Skill Coverage (DWSC) ────────────────────────────
    hard_covered = sum(v for k, v in hard_job_counts.items() if k in course_skill_set)
    soft_covered = sum(v for k, v in soft_job_counts.items() if k in course_skill_set)
    dwsc_hard = hard_covered / hard_demand_total if hard_demand_total else 0.0
    dwsc_soft = soft_covered / soft_demand_total if soft_demand_total else 0.0

    # ── Two-pass skill classification ─────────────────────────────────────
    course_hard_mentions = 0
    course_soft_mentions = 0
    course_unknown_mentions = 0
    type_rows = []

    for skill, mentions in course_skill_counts.items():
        skill_type = classify_skill(skill, hard_job_counts, soft_job_counts)
        if skill_type == "hard":
            course_hard_mentions += int(mentions)
        elif skill_type == "soft":
            course_soft_mentions += int(mentions)
        else:
            course_unknown_mentions += int(mentions)

        type_rows.append({
            "skill":             skill,
            "course_mentions":   int(mentions),
            "job_hard_demand":   int(hard_job_counts.get(skill, 0)),
            "job_soft_demand":   int(soft_job_counts.get(skill, 0)),
            "assigned_type":     skill_type,
            # Record which pass made the classification for transparency
            "classification_source": (
                "lexicon" if skill in _LEXICON_MAP else
                "job_demand" if (hard_job_counts.get(skill, 0) + soft_job_counts.get(skill, 0)) > 0
                else "unknown"
            ),
        })

    typed_total = course_hard_mentions + course_soft_mentions
    course_hard_share = course_hard_mentions / typed_total if typed_total else 0.0
    course_soft_share = course_soft_mentions / typed_total if typed_total else 0.0
    jobs_hard_share = hard_demand_total / total_job_demand if total_job_demand else 0.0
    jobs_soft_share = soft_demand_total / total_job_demand if total_job_demand else 0.0

    hard_share_gap = course_hard_share - jobs_hard_share
    soft_share_gap = course_soft_share - jobs_soft_share

    # ── Uncovered job skills ──────────────────────────────────────────────
    uncovered_rows = []
    for skill, count in hard_job_counts.items():
        if skill not in course_skill_set:
            uncovered_rows.append({"skill": skill, "type": "hard",
                                   "job_demand_count": int(count)})
    for skill, count in soft_job_counts.items():
        if skill not in course_skill_set:
            uncovered_rows.append({"skill": skill, "type": "soft",
                                   "job_demand_count": int(count)})

    uncovered_df = pd.DataFrame(uncovered_rows).sort_values(
        by=["type", "job_demand_count"], ascending=[True, False]
    )

    # ── Assemble metrics dict ─────────────────────────────────────────────
    metrics = {
        "inputs": {
            "module_skill_pairs_path": str(module_skill_pairs_path),
            "jobs_skills_path":        str(jobs_skills_path),
            "module_skill_rows":       int(len(modules_df)),
            "job_rows":                int(len(jobs_df)),
            "unique_course_skills":    int(len(course_skill_set)),
            "unique_hard_job_skills":  int(len(hard_job_counts)),
            "unique_soft_job_skills":  int(len(soft_job_counts)),
        },
        "coverage_metrics": {
            "dwsc_hard":           round(dwsc_hard, 6),
            "dwsc_soft":           round(dwsc_soft, 6),
            "hard_job_demand_total": int(hard_demand_total),
            "soft_job_demand_total": int(soft_demand_total),
            "hard_demand_covered": int(hard_covered),
            "soft_demand_covered": int(soft_covered),
        },
        "emphasis_metrics": {
            "course_hard_share":   round(course_hard_share, 6),
            "course_soft_share":   round(course_soft_share, 6),
            "jobs_hard_share":     round(jobs_hard_share, 6),
            "jobs_soft_share":     round(jobs_soft_share, 6),
            "hard_share_gap_course_minus_jobs": round(hard_share_gap, 6),
            "soft_share_gap_course_minus_jobs": round(soft_share_gap, 6),
            "course_hard_mentions":    int(course_hard_mentions),
            "course_soft_mentions":    int(course_soft_mentions),
            "course_unknown_mentions": int(course_unknown_mentions),
            # v2: report unknown fraction explicitly so metric consumers
            # know the share denominators are partial
            "unknown_fraction_of_course_mentions": round(
                course_unknown_mentions /
                max(course_hard_mentions + course_soft_mentions + course_unknown_mentions, 1),
                6,
            ),
        },
        "notes": {
            "classifier": (
                "Two-pass: (1) curated lexicon of ~120 hard/soft seed skills; "
                "(2) job-demand count fallback for skills not in lexicon. "
                "Skills absent from both are marked unknown and excluded from "
                "share denominators. See unknown_fraction_of_course_mentions."
            ),
            "canonicalization": (
                "Both course and job skills are singularized and synonym-mapped "
                "before matching, reducing fragmentation artefacts "
                "(e.g. 'algorithms' == 'algorithm')."
            ),
        },
    }

    # ── Level-segmented emphasis metrics ─────────────────────────────────
    # Addresses the core limitation: job-alignment is not the only measure
    # of a course's value.  Foundational modules are expected to be broad;
    # advanced modules should show stronger vocational alignment.
    # We compute hard/soft share gap separately for each level so you can
    # show that the gap is concentrated in intermediate/advanced modules
    # (a meaningful finding) rather than foundational ones (expected).

    modules_df["module_level"] = modules_df["module code"].astype(str).map(
        get_module_level
    )

    level_metrics: dict[str, dict] = {}
    for level in ("foundational", "intermediate", "advanced", "unknown"):
        level_df = modules_df[modules_df["module_level"] == level]
        if level_df.empty:
            continue

        level_skill_counts = level_df["skill_canon"].value_counts()
        lv_hard = lv_soft = lv_unknown = 0

        for skill, mentions in level_skill_counts.items():
            t = classify_skill(skill, hard_job_counts, soft_job_counts)
            if t == "hard":
                lv_hard += int(mentions)
            elif t == "soft":
                lv_soft += int(mentions)
            else:
                lv_unknown += int(mentions)

        lv_typed = lv_hard + lv_soft
        lv_hard_share = lv_hard / lv_typed if lv_typed else 0.0
        lv_soft_share = lv_soft / lv_typed if lv_typed else 0.0

        level_metrics[level] = {
            "module_count":      int(level_df["module code"].nunique()),
            "hard_mentions":     lv_hard,
            "soft_mentions":     lv_soft,
            "unknown_mentions":  lv_unknown,
            "hard_share":        round(lv_hard_share, 6),
            "soft_share":        round(lv_soft_share, 6),
            # Gap vs job market — the key number to discuss with MOE:
            # a large gap in 'advanced' modules is more concerning than
            # the same gap in 'foundational' modules.
            "hard_share_gap_vs_jobs": round(lv_hard_share - jobs_hard_share, 6),
            "soft_share_gap_vs_jobs": round(lv_soft_share - jobs_soft_share, 6),
        }

    metrics["level_segmented_emphasis"] = level_metrics
    metrics["notes"]["level_segmentation"] = (
        "Module level inferred from first digit of module code: "
        "1000-2000 = foundational, 3000-4000 = intermediate, 5000+ = advanced. "
        "Foundational modules are expected to have a larger hard/soft gap because "
        "breadth education is their purpose, not vocational alignment. "
        "The gap in intermediate and advanced modules is the policy-relevant signal."
    )

    # ── Build output DataFrames ───────────────────────────────────────────
    skill_types_df = pd.DataFrame(type_rows).sort_values(
        by=["assigned_type", "course_mentions"], ascending=[True, False]
    )

    module_skill_types_df = (
        modules_df
        .merge(
            skill_types_df[["skill", "assigned_type", "classification_source"]],
            left_on="skill_canon",
            right_on="skill",
            how="left",
            suffixes=("", "_typed"),
        )
        .rename(columns={"assigned_type": "skill_type"})
    )
    module_skill_types_df["skill_type"] = (
        module_skill_types_df["skill_type"].fillna("unknown")
    )
    module_skill_types_df = module_skill_types_df[
        ["module code", "title", "skill", "skill_type",
         "classification_source", "module_level"]
    ].drop_duplicates()

    return metrics, uncovered_df, skill_types_df, module_skill_types_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute hard/soft skill-gap metrics between courses and jobs."
    )
    parser.add_argument(
        "--module-skill-pairs",
        default="data/processed/module_skill_pairs.csv",
        help="Path to module-skill pair CSV (one row per module-skill).",
    )
    parser.add_argument(
        "--jobs-skills",
        default="data/processed/jobs_processed_with_skills_relevant.csv",
        help="Path to jobs CSV with hard_skills / soft_skills columns.",
    )
    parser.add_argument(
        "--summary-out",
        default="data/processed/hard_soft_skill_gap_metrics.json",
    )
    parser.add_argument(
        "--uncovered-out",
        default="data/processed/uncovered_job_skills_by_type.csv",
    )
    parser.add_argument(
        "--course-types-out",
        default="data/processed/course_skill_type_assignment.csv",
    )
    parser.add_argument(
        "--module-types-out",
        default="data/processed/modules_with_skill_types.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, uncovered_df, skill_types_df, module_skill_types_df = compute_metrics(
        module_skill_pairs_path=Path(args.module_skill_pairs),
        jobs_skills_path=Path(args.jobs_skills),
    )

    for path in [args.summary_out, args.uncovered_out,
                 args.course_types_out, args.module_types_out]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    uncovered_df.to_csv(args.uncovered_out, index=False)
    skill_types_df.to_csv(args.course_types_out, index=False)
    module_skill_types_df.to_csv(args.module_types_out, index=False)

    print(f"Saved summary metrics  : {args.summary_out}")
    print(f"Saved uncovered skills  : {args.uncovered_out}")
    print(f"Saved course skill types: {args.course_types_out}")
    print(f"Saved module skill types: {args.module_types_out}")
    print()
    em = metrics["emphasis_metrics"]
    cm = metrics["coverage_metrics"]
    print(f"DWSC hard / soft        : {cm['dwsc_hard']:.4f} / {cm['dwsc_soft']:.4f}")
    print(f"Course hard/soft share  : {em['course_hard_share']:.3f} / {em['course_soft_share']:.3f}")
    print(f"Jobs   hard/soft share  : {em['jobs_hard_share']:.3f} / {em['jobs_soft_share']:.3f}")
    print(f"Gap (course - jobs)     : hard {em['hard_share_gap_course_minus_jobs']:+.3f} "
          f"/ soft {em['soft_share_gap_course_minus_jobs']:+.3f}")
    print(f"Unknown skill fraction  : {em['unknown_fraction_of_course_mentions']:.3f}")
    print()
    print("Hard/soft gap by module level (course share - jobs share):")
    for level, lm in metrics["level_segmented_emphasis"].items():
        print(
            f"  {level:15s}  n={lm['module_count']:>5}  "
            f"hard_gap={lm['hard_share_gap_vs_jobs']:+.3f}  "
            f"soft_gap={lm['soft_share_gap_vs_jobs']:+.3f}"
        )


if __name__ == "__main__":
    main()
