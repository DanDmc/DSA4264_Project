import argparse
import re
import sys
from pathlib import Path

import pandas as pd

# Local fallback keywords for environments where SkillNER is unavailable.
FALLBACK_SKILLS = [
    "python",
    "r programming",
    "stata",
    "spss",
    "excel",
    "sql",
    "statistics",
    "biostatistics",
    "data analysis",
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "nlp",
    "natural language processing",
    "computer vision",
    "research",
    "communication",
    "leadership",
    "project management",
    "teamwork",
    "problem solving",
    "critical thinking",
    "public speaking",
    "writing",
    "academic writing",
    "presentation",
    "econometrics",
    "finance",
    "accounting",
    "marketing",
    "entrepreneurship",
    "design thinking",
    "ui design",
    "ux design",
    "software engineering",
    "web development",
    "java",
    "c++",
    "javascript",
    "react",
    "node.js",
    "git",
]


class KeywordSkillExtractor:
    def __init__(self, skills: list[str]) -> None:
        self._compiled_patterns: list[tuple[str, re.Pattern[str]]] = []
        for skill in skills:
            escaped = re.escape(skill.lower())
            pattern = re.compile(r"(?<!\w)" + escaped + r"(?!\w)", flags=re.IGNORECASE)
            self._compiled_patterns.append((skill, pattern))

        # Additional aliases for common abbreviations.
        self._compiled_patterns.append(("r programming", re.compile(r"(?<!\w)R(?!\w)")))

    def annotate(self, text: str) -> dict:
        matches = []
        for skill, pattern in self._compiled_patterns:
            if pattern.search(text):
                matches.append({"doc_node_value": skill})
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
        return KeywordSkillExtractor(FALLBACK_SKILLS)


def _coerce_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


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

    extractor = build_extractor()
    rows = []
    module_code_idx = df.columns.get_loc("module code")
    title_idx = df.columns.get_loc("title")
    description_idx = df.columns.get_loc("description")

    for row in df.itertuples(index=False, name=None):
        description = _coerce_text(row[description_idx])
        if not description:
            skills = []
        else:
            annotation = extractor.annotate(description)
            skills = extract_skill_names(annotation)

        rows.append(
            {
                "module code": row[module_code_idx],
                "title": row[title_idx],
                "description": description,
                "skills": ", ".join(skills),
                "skill_count": len(skills),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract skills from NUSMods module descriptions."
    )
    parser.add_argument(
        "--input",
        default="modules.csv",
        help="Path to input CSV from NUSMods (default: modules.csv).",
    )
    parser.add_argument(
        "--output-rows",
        default="modules_with_skills.csv",
        help="Output CSV with one row per module and aggregated skills.",
    )
    parser.add_argument(
        "--output-skills",
        default="module_skill_pairs.csv",
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
    print(f"Saved: {args.output_rows}")
    print(f"Saved: {args.output_skills}")


if __name__ == "__main__":
    main()
