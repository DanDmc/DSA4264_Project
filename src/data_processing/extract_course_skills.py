import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Canonical skill taxonomy with aliases, tuned for mixed NUS module domains.
FALLBACK_SKILL_TAXONOMY = {
    "python": ["python"],
    "r programming": ["r programming", " r ", "(r)", "r language"],
    "stata": ["stata"],
    "spss": ["spss"],
    "excel": ["excel", "ms excel", "microsoft excel"],
    "sql": ["sql"],
    "data analysis": ["data analysis", "data analytics", "analytical methods"],
    "statistics": ["statistics", "statistical"],
    "biostatistics": ["biostatistics", "biostatistical"],
    "machine learning": ["machine learning", "ml"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "deep learning": ["deep learning"],
    "natural language processing": ["natural language processing", "nlp"],
    "computer vision": ["computer vision"],
    "software engineering": ["software engineering"],
    "web development": ["web development"],
    "java": ["java"],
    "c++": ["c++"],
    "javascript": ["javascript", "js"],
    "react": ["react"],
    "node.js": ["node.js", "nodejs"],
    "git": ["git", "version control"],
    "project management": ["project management", "programme management"],
    "leadership": ["leadership", "lead"],
    "communication": ["communication", "communicate"],
    "presentation": ["presentation", "presenting", "pitch"],
    "critical thinking": ["critical thinking"],
    "problem solving": ["problem solving", "problem-solving"],
    "teamwork": ["teamwork", "team-based", "team work"],
    "academic writing": ["academic writing"],
    "writing": ["writing", "written communication"],
    "research methods": ["research methods", "methodology"],
    "quantitative research": ["quantitative research"],
    "qualitative research": ["qualitative research"],
    "econometrics": ["econometrics", "econometric"],
    "finance": ["finance", "financial analysis"],
    "accounting": ["accounting"],
    "marketing": ["marketing"],
    "entrepreneurship": ["entrepreneurship", "venture creation"],
    "design thinking": ["design thinking", "human-centered design"],
    "ui design": ["ui design", "user interface design"],
    "ux design": ["ux design", "user experience design"],
    "architecture": ["architecture", "architectural"],
    "urban planning": ["urban planning", "city planning"],
    "conservation": ["conservation", "heritage conservation"],
    "heritage management": ["heritage management", "cultural heritage"],
    "risk management": ["risk management", "disaster risk"],
    "immunology": ["immunology", "immunological"],
    "microbiology": ["microbiology", "microbiome"],
    "pharmacology": ["pharmacology", "drug development"],
    "vaccine development": ["vaccine development", "vaccinology"],
    "public health": ["public health", "health policy", "epidemiology"],
}


class KeywordSkillExtractor:
    def __init__(self, taxonomy: dict[str, list[str]]) -> None:
        self._compiled_patterns: list[tuple[str, re.Pattern[str]]] = []
        for canonical_skill, aliases in taxonomy.items():
            for alias in aliases:
                escaped = re.escape(alias.lower().strip())
                pattern = re.compile(
                    r"(?<!\w)" + escaped + r"(?!\w)",
                    flags=re.IGNORECASE,
                )
                self._compiled_patterns.append((canonical_skill, pattern))

    def annotate(self, text: str) -> dict:
        matches = []
        for canonical_skill, pattern in self._compiled_patterns:
            if pattern.search(text):
                matches.append({"doc_node_value": canonical_skill})
        return {"results": {"full_matches": matches, "ngram_scored": []}}


def build_extractor():
    """Build SkillNER extractor; fallback to keyword matcher if unavailable."""
    try:
        import spacy
        from spacy.matcher import PhraseMatcher

        from requests.exceptions import RequestException
        from skillNer.general_params import SKILL_DB
        from skillNer.skill_extractor_class import SkillExtractor

        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            nlp = spacy.load("en_core_web_sm")
        print("Using extractor backend: SkillNER", file=sys.stderr)
        return SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    except (ImportError, ModuleNotFoundError, OSError, RequestException) as exc:
        print(
            f"SkillNER unavailable ({type(exc).__name__}: {exc}). "
            "Falling back to keyword extractor.",
            file=sys.stderr,
        )
        print("Using extractor backend: keyword fallback", file=sys.stderr)
        return KeywordSkillExtractor(FALLBACK_SKILL_TAXONOMY)


def _coerce_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _is_valid_description(text: str) -> bool:
    """Keep only rows with meaningful module descriptions."""
    if not text:
        return False
    normalized = text.strip().lower().replace("\u2018", "'").replace("\u2019", "'")

    # Approved placeholder removals for reproducible filtering.
    if normalized in {
        "not available",
        "not available.",
        "not applicable",
        "unrestricted elective",
        "nil",
        "department exchange course",
        "advance placement credit",
    }:
        return False

    # Exchange placeholder descriptions (keep rule narrow and explicit).
    if re.fullmatch(r"(faculty )?exchange course( - yus \(1 unit\))?", normalized):
        return False

    return True


def clean_existing_outputs(output_rows_csv: Path, output_skills_csv: Path) -> None:
    """Clean already-generated outputs in place using current description rules."""
    if not output_rows_csv.exists():
        return

    rows_df = pd.read_csv(output_rows_csv)
    if {"module code", "description"} - set(rows_df.columns):
        return

    original_rows = len(rows_df)
    desc_series = rows_df["description"].apply(_coerce_text)
    rows_df = rows_df[desc_series.apply(_is_valid_description)].copy()
    rows_df.to_csv(output_rows_csv, index=False)

    if output_skills_csv.exists():
        pairs_df = pd.read_csv(output_skills_csv)
        if "module code" in pairs_df.columns:
            original_pairs = len(pairs_df)
            valid_codes = set(rows_df["module code"].astype(str))
            pairs_df = pairs_df[pairs_df["module code"].astype(str).isin(valid_codes)]
            pairs_df.to_csv(output_skills_csv, index=False)
            print(
                "Cleaned existing outputs: "
                f"{output_rows_csv.name} {original_rows}->{len(rows_df)}, "
                f"{output_skills_csv.name} {original_pairs}->{len(pairs_df)}",
                file=sys.stderr,
            )
            return

    print(
        "Cleaned existing outputs: "
        f"{output_rows_csv.name} {original_rows}->{len(rows_df)}",
        file=sys.stderr,
    )


def extract_skill_names(annotation: dict) -> list[str]:
    """Collect skill labels from SkillNER output with defensive parsing."""
    results = annotation.get("results", {})
    candidates = []

    for key in ("full_matches", "ngram_scored"):
        for item in results.get(key, []):
            label = (
                item.get("doc_node_value")
                or item.get("skill_name")
                or item.get("skill")
                or item.get("text")
            )
            if label:
                candidates.append(str(label).strip())

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def extract_course_skills(
    input_csv: Path,
    output_rows_csv: Path,
    output_skills_csv: Path,
) -> None:
    df = pd.read_csv(input_csv)
    required = {"module code", "title", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    # Drop rows with missing placeholder descriptions before extraction.
    original_count = len(df)
    desc_series = df["description"].apply(_coerce_text)
    df = df[desc_series.apply(_is_valid_description)].copy()
    dropped_count = original_count - len(df)
    if dropped_count:
        print(
            f"Filtered out {dropped_count} module(s) with unavailable descriptions.",
            file=sys.stderr,
        )

    extractor = build_extractor()
    keyword_fallback = KeywordSkillExtractor(FALLBACK_SKILL_TAXONOMY)
    rows = []
    module_code_idx = df.columns.get_loc("module code")
    title_idx = df.columns.get_loc("title")
    description_idx = df.columns.get_loc("description")
    row_fallback_count = 0

    for row in df.itertuples(index=False, name=None):
        title = _coerce_text(row[title_idx])
        description = _coerce_text(row[description_idx])
        description_skills = []
        if description:
            try:
                description_annotation = extractor.annotate(description)
            except Exception as exc:
                row_fallback_count += 1
                if row_fallback_count <= 5:
                    print(
                        "SkillNER row-level failure "
                        f"({type(exc).__name__}). "
                        "Using keyword fallback for this row.",
                        file=sys.stderr,
                    )
                description_annotation = keyword_fallback.annotate(description)
            description_skills = extract_skill_names(description_annotation)

        rows.append(
            {
                "module code": row[module_code_idx],
                "title": title,
                "description": description,
                "skills": ", ".join(description_skills),
                "skill_count": len(description_skills),
            }
        )

    output_rows = pd.DataFrame(rows)
    output_rows.to_csv(output_rows_csv, index=False)

    exploded = output_rows.copy()
    exploded["skills"] = exploded["skills"].str.split(", ")
    exploded = exploded.explode("skills")
    exploded = exploded[exploded["skills"].notna() & (exploded["skills"] != "")]
    exploded = exploded.rename(columns={"skills": "skill"})
    exploded = exploded[["module code", "title", "skill"]]
    exploded = exploded.drop_duplicates()
    exploded.to_csv(output_skills_csv, index=False)
    if row_fallback_count:
        print(
            f"Row-level keyword fallback used for {row_fallback_count} module(s).",
            file=sys.stderr,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract skills from NUSMods module descriptions."
    )
    parser.add_argument(
        "--input",
        default="data/processed/cleaned_modules.csv",
        help="Path to input CSV from NUSMods (default: data/processed/cleaned_modules.csv).",
    )
    parser.add_argument(
        "--output-rows",
        default="data/processed/modules_with_skills.csv",
        help="Output CSV with one row per module and aggregated skills.",
    )
    parser.add_argument(
        "--output-skills",
        default="data/processed/module_skill_pairs.csv",
        help="Output CSV with one row per module-skill pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_course_skills(
        input_csv=Path(args.input),
        output_rows_csv=Path(args.output_rows),
        output_skills_csv=Path(args.output_skills),
    )
    clean_existing_outputs(
        output_rows_csv=Path(args.output_rows),
        output_skills_csv=Path(args.output_skills),
    )
    print(f"Saved: {args.output_rows}")
    print(f"Saved: {args.output_skills}")


if __name__ == "__main__":
    main()
