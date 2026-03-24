"""
postprocess_skills.py
=====================
Cleans and canonicalizes the raw SkillNer skill output before the
hard/soft classification step.
 
Why this script exists
----------------------
SkillNer was trained on job postings, not academic text.  Even with
the description preprocessing in process_nusmods.py it still produces:
 
  1. Noise tokens  -- single characters ('e'), course-logistics words
     ('lectures', 'readings'), generic verbs used as nouns ('targeted',
     'illustrated'), and topic words that are not transferable skills.
 
  2. Plural/inflected surface forms  -- 'algorithms', 'presentations',
     'communications' treated as different skills from 'algorithm',
     'presentation', 'communication', which inflates the unique-skill
     count and suppresses coverage metrics.
 
  3. Near-duplicate fragments  -- 'algorithm design', 'design algorithms',
     'algorithms design' all appearing as distinct entries.
 
This script fixes all three problems before the data reaches
hard_soft_skills.py.
 
Usage
-----
    python postprocess_skills.py \
        --input  modules_with_skills_raw.csv \
        --output modules_with_skills.csv
 
Input CSV columns required
--------------------------
    module code, title, description, skills, skill_count
 
    `skills` must be either:
      * a JSON list string:  '["python", "data analysis"]'   (preferred)
      * a comma-separated string:  'python, data analysis'   (legacy SkillNer)
 
Output CSV columns
------------------
    Same as input plus `skills_raw` (original) and updated `skills` /
    `skill_count` after cleaning.
 
Requirements
------------
    pip install pandas inflect
"""
 
import argparse
import json
import re
from pathlib import Path
 
import inflect
import pandas as pd
 
# ---------------------------------------------------------------------------
# Noise stoplist
# Words / phrases that SkillNer extracts but are NOT transferable skills.
# Add to this list whenever you spot new noise in the output.
# ---------------------------------------------------------------------------
_STOPLIST: frozenset[str] = frozenset({
    # ── Course logistics ──────────────────────────────────────────────────
    "lecture", "lectures",
    "reading", "readings",
    "tutorial", "tutorials",
    "seminar", "seminars",
    "workshop", "workshops",
    "recitation",
    "independent study",
    "case study", "case studies",
    "students learn", "working environment",
 
    # ── Generic verbs returned as noun-like tokens ────────────────────────
    "communicate", "investigate", "reach", "furnish", "targeted",
    "illustrated", "supervised", "translate", "integrated", "modulate",
    "discussed", "explore", "examine", "read", "write",
    "presented", "governing", "integrate", "operate", "illustrate",
    "manage", "teach", "hone", "dive",
 
    # ── Topics / fields (studied, not practised as a skill) ───────────────
    "social", "economic", "economics", "urban", "government", "economy",
    "construction", "governance", "sustainability", "teaching", "supervision",
    "perspective", "perspectives", "conceptual", "integration", "interaction",
    "transformation", "innovative", "creative", "managing",
    "debates", "debate", "inquiry", "investigation",
 
    # ── SkillNer-specific artefacts ───────────────────────────────────────
    "program generate", "track", "stage", "msc",
    "decision make",   # artefact of "decision making"
    "problem solve",   # artefact of "problem solving"
    "lecturer", "manager", "target", "claim",
    "basic principle", "experiential learning",
 
    # ── Garbage single tokens ─────────────────────────────────────────────
    "e", "b", "m", "c", "ar", "vis", "ph", "ad", "rna",
    "2d", "3d", "act", "ear", "food", "film", "jazz",
    "host", "roof", "arab", "maps", "map",
})
 
# Words that end in -s but are NOT plurals and must NOT be singularized.
_PROTECT_SINGULARIZE: frozenset[str] = frozenset({
    "ethics", "economics", "statistics", "mathematics", "physics", "politics",
    "logistics", "linguistics", "genetics", "mechanics", "dynamics",
    "electronics", "robotics", "hydraulics", "acoustics", "thermodynamics",
    "analysis", "basis", "thesis", "axis", "crisis", "series", "species",
    "news", "data", "media", "criteria", "phenomena", "corpus",
    # domain-specific
    "bioinformatics", "informatics", "forensics",
})
 
# Manual synonym / alias map applied AFTER singularization.
# Left side = noisy SkillNer form, right side = canonical form.
# Extend this whenever you find clusters of near-duplicate skills.
_SYNONYM_MAP: dict[str, str] = {
    # algorithm variants
    "algorithmic":          "algorithm design",
    "algorithms":           "algorithm",          # handled by singularizer too
    "designing algorithm":  "algorithm design",
    "algorithms design":    "algorithm design",
    "algorithms search":    "search algorithm",
    "machine algorithm":    "machine learning",
    "machine algorithms":   "machine learning",
    "machine learn algorithm": "machine learning",
 
    # analytics / analytical
    "analytic":             "analytical thinking",
    "analytical":           "analytical thinking",
 
    # writing variants
    "write":                "writing",
    "writings":             "writing",
 
    # presentation variants
    "present":              "presentation",
    "presentations":        "presentation",
 
    # communication variants
    "communicating":        "communication",
    "communications":       "communication",
 
    # research variants
    "researcher":           "research",
    "researching":          "research",
 
    # decision making
    "decision make":        "decision making",
 
    # problem solving
    "problem solve":        "problem solving",
    "problem-solving":      "problem solving",
 
    # data analysis
    "data analyses":        "data analysis",
 
    # modelling
    "modeling":             "modelling",
    "financial modeling":   "financial modelling",
    "predictive modeling":  "predictive modelling",
    "3d modeling":          "3d modelling",
}
 
# ---------------------------------------------------------------------------
# Inflect engine (module-level singleton)
# ---------------------------------------------------------------------------
_INFLECT = inflect.engine()
 
 
# ---------------------------------------------------------------------------
# Core normalization helpers
# ---------------------------------------------------------------------------
 
def _singularize_last_word(skill: str) -> str:
    """Singularize the last word of a skill phrase unless it is protected."""
    words = skill.split()
    if not words:
        return skill
    last = words[-1]
    if last in _PROTECT_SINGULARIZE:
        return skill
    singular = _INFLECT.singular_noun(last)
    if singular:
        words[-1] = singular
    return " ".join(words)
 
 
def canonicalize(skill: str) -> str:
    """Return the canonical lowercase, singularized form of a skill string.
 
    Pipeline:
      1. Lowercase + strip whitespace
      2. Singularize last word (protected nouns excluded)
      3. Apply synonym / alias map
    """
    skill = " ".join(skill.strip().lower().split())  # normalise whitespace
    skill = _singularize_last_word(skill)
    skill = _SYNONYM_MAP.get(skill, skill)
    return skill
 
 
def is_valid(skill: str) -> bool:
    """Return True if the skill should be kept after canonicalization.
 
    Rejects:
      - Strings of 2 characters or fewer
      - Entries in the noise stoplist
      - Pure numeric strings
      - Single tokens that are 4 chars or shorter AND not an acronym
        (catches 'case', 'read', 'hone', 'dive', '2d', etc.)
    """
    s = skill.strip().lower()
    if len(s) <= 2:
        return False
    if s in _STOPLIST:
        return False
    if s.replace(" ", "").isdigit():
        return False
    words = s.split()
    if len(words) == 1 and len(s) <= 4 and not s.isupper():
        return False
    return True
 
 
# ---------------------------------------------------------------------------
# Skill list parser  (handles both JSON-list and comma-separated strings)
# ---------------------------------------------------------------------------
 
def parse_skill_list(value: object) -> list[str]:
    """Parse a skills cell that may be JSON or comma-separated."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
 
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
 
    # Try JSON first (preferred format from extract_skills.py)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except json.JSONDecodeError:
        pass
 
    # Fallback: comma-separated (legacy SkillNer output)
    return [x.strip() for x in text.split(",") if x.strip()]
 
 
# ---------------------------------------------------------------------------
# Row-level processing
# ---------------------------------------------------------------------------
 
def clean_skill_list(raw: list[str]) -> list[str]:
    """Canonicalize and filter a single module's skill list."""
    seen: set[str] = set()
    result: list[str] = []
    for skill in raw:
        canonical = canonicalize(skill)
        if is_valid(canonical) and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result
 
 
# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
 
def run(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
 
    required = {"module code", "skills"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
 
    # Keep a copy of the raw SkillNer output for auditability
    df["skills_raw"] = df["skills"]
 
    cleaned_skills: list[str] = []
    cleaned_counts: list[int] = []
 
    for raw_val in df["skills"]:
        raw_list = parse_skill_list(raw_val)
        clean_list = clean_skill_list(raw_list)
        # Store as JSON list string for clean round-tripping into hard_soft_skills.py
        cleaned_skills.append(json.dumps(clean_list))
        cleaned_counts.append(len(clean_list))
 
    df["skills"] = cleaned_skills
    df["skill_count"] = cleaned_counts
 
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
 
    # ── Summary stats ────────────────────────────────────────────────────
    total_raw = sum(
        len(parse_skill_list(v)) for v in df["skills_raw"]
    )
    total_clean = sum(cleaned_counts)
    all_clean = [
        s
        for v in df["skills"]
        for s in parse_skill_list(v)
    ]
 
    print(f"Modules processed : {len(df):,}")
    print(f"Total skill tokens: {total_raw:,} -> {total_clean:,}  "
          f"(removed {total_raw - total_clean:,}, "
          f"{100 * (total_raw - total_clean) / max(total_raw, 1):.1f}%)")
    print(f"Unique skills     : {len(set(all_clean)):,}  "
          f"(was {len(set(s.strip().lower() for v in df['skills_raw'] for s in parse_skill_list(v))):,})")
    print(f"Output saved to   : {output_path}")
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and canonicalize SkillNer skill output."
    )
    parser.add_argument(
        "--input",
        default="modules_with_skills_raw.csv",
        help="Raw SkillNer output CSV (default: modules_with_skills_raw.csv)",
    )
    parser.add_argument(
        "--output",
        default="modules_with_skills.csv",
        help="Cleaned output CSV (default: modules_with_skills.csv)",
    )
    return parser.parse_args()
 
 
def main() -> None:
    args = parse_args()
    run(Path(args.input), Path(args.output))
 
 
if __name__ == "__main__":
    main()
 