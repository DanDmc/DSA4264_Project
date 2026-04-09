"""
extract_course_skills_llm.py
=============================
Production LLM skill extractor for all NUS course descriptions.

Uses the same method-3 (LLM few-shot) prompt that was validated in
src/validation/compare_extractors.py (F1 = 0.580 on 39 annotated modules).

Reads  : data/processed/cleaned_modules.csv
Writes : data/processed/module_skill_pairs.csv   (one row per module-skill pair)

The knowledge_type column produced here is a first-pass label from the
extraction prompt.  Run classify_skills.py afterwards to replace it with
the dedicated classifier's labels.

LLM providers are tried in order: cerebras → groq → gemini → ollama.
See src/llm_provider.py for configuration.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

from src.llm_provider import call_llm, available_providers


# ---------------------------------------------------------------------------
# Extraction prompt  (frozen — matches compare_extractors.py method 3)
# ---------------------------------------------------------------------------

_SKILL_DEFINITION = (
    "A skill is something the text explicitly states the course teaches, "
    "specific enough to represent a distinct concept a student could have or not have."
)

_ANNOTATION_RULES = """\
RULES:
- Only extract skills EXPLICITLY stated in the text. Do NOT infer or imply.
- Include technical terms AND academic/domain terms (e.g. "dynamic programming", "convex optimisation").
- Do NOT include generic framing (e.g. "various topics", "broad overview", "applications").
- skill_type: "hard" = technical / domain knowledge.  "soft" = communication, teamwork, etc.
- knowledge_type:
    "applied"      — action verbs in context: implement, build, design, derive, select, verify, analyse
    "theoretical"  — conceptual verbs or topic listings: understand, covers, introduces, examines
- "understanding of X" framing → X is theoretical.
- Listing topics without action verbs ("topics include X, Y, Z") → all theoretical."""

_SYSTEM_PROMPT = f"""You are a skill extraction assistant for NUS university course descriptions.

SKILL DEFINITION: {_SKILL_DEFINITION}

{_ANNOTATION_RULES}

OUTPUT: Return ONLY a valid JSON array. No explanation text. No markdown.
Example element: {{"skill_label": "dynamic programming", "skill_type": "hard", "knowledge_type": "applied"}}"""

# Few-shot examples — trimmed to 3 representative cases to reduce token usage
# per call (~400 tokens vs ~800 for the full 6-example set).
# Covers: mixed theoretical/applied, zero-skill, all-theoretical topic listing.
_FEW_SHOT_EXAMPLES: list[dict] = [
    {
        # Shows: "covered from viewpoint of" → theoretical; active tasks → applied
        "description": (
            "Financial reporting is covered from the viewpoint of an external investor. "
            "Book-keeping and preparation of financial statements are also covered."
        ),
        "skills": [
            {"skill_label": "financial reporting",                "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "book-keeping",                       "skill_type": "hard", "knowledge_type": "applied"},
            {"skill_label": "preparation of financial statements","skill_type": "hard", "knowledge_type": "applied"},
        ],
    },
    {
        # Shows: zero-skill — approach descriptors are NOT taught skills
        "description": (
            "This module introduces students to a data-driven, innovative systems approach to "
            "engineering design. Students will work in teams to develop novel solutions."
        ),
        "skills": [],
    },
    {
        # Shows: topic listings with "They include" → all theoretical
        "description": (
            "Several statistical methodologies useful for exploring high-dimensional data. "
            "They include principal component analysis, clustering, neural networks, "
            "and penalized estimation for variable selection."
        ),
        "skills": [
            {"skill_label": "principal component analysis", "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "clustering",                   "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "neural networks",              "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "penalized estimation",         "skill_type": "hard", "knowledge_type": "theoretical"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Prompt builder and response parser
# ---------------------------------------------------------------------------

def _build_prompt(description: str) -> str:
    parts = []
    for ex in _FEW_SHOT_EXAMPLES:
        parts.append(
            f"Course description:\n{ex['description']}\n\n"
            f"Extracted skills:\n{json.dumps(ex['skills'], indent=2)}"
        )
    parts.append(f"Course description:\n{description}\n\nExtracted skills:")
    return "\n\n---\n\n".join(parts)


_PUNC_RE = re.compile(r"[^\w\s\-]")


def _parse_response(text: str) -> list[dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return []


def _to_skill_rows(raw: list[dict], module_code: str, title: str) -> list[dict]:
    rows = []
    seen: set[str] = set()
    for item in raw:
        label = str(item.get("skill_label", "")).strip().lower()
        if not label or label in seen:
            continue
        seen.add(label)
        rows.append({
            "module_code":    module_code,
            "title":          title,
            "skill_label":    label,
            "skill_type":     str(item.get("skill_type", "unknown")).lower(),
            "knowledge_type": str(item.get("knowledge_type", "unknown")).lower(),
        })
    return rows


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def _save(rows: list[dict], path: Path) -> None:
    out = pd.DataFrame(rows, columns=["module_code", "title", "skill_label",
                                      "skill_type", "knowledge_type"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def extract_all(
    input_csv: Path,
    output_pairs_csv: Path,
    providers: list[str] | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> None:
    """Extract skills from all (or a slice of) modules in input_csv.

    Args:
        start_idx: First row index to process (0-based). Use to split work
                   across groupmates: person A gets 0–2000, person B 2000–4000.
        end_idx:   Exclusive end row index. None = process to the last row.

    Checkpoint/resume: if output_pairs_csv already exists, modules already
    present in it are skipped. Re-run after a daily quota reset to continue.
    """
    df = pd.read_csv(input_csv)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "module_code" not in df.columns and "module" in df.columns:
        df = df.rename(columns={"module": "module_code"})

    desc_col = next(
        (c for c in ("description_clean", "description") if c in df.columns), None
    )
    if desc_col is None:
        raise ValueError(f"No description column in {input_csv}. Columns: {list(df.columns)}")

    title_col = "title" if "title" in df.columns else None

    # Apply row slice (for splitting work across groupmates)
    total = len(df)
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    if start_idx or end_idx:
        print(f"[INFO] Processing rows {start_idx}–{end_idx or total} "
              f"({len(df)} of {total} modules).", file=sys.stderr)

    if providers is None:
        providers = available_providers()
    if not providers:
        raise RuntimeError(
            "No LLM providers available. Set CEREBRAS_API_KEY, GROQ_API_KEY, "
            "GEMINI_API_KEY, or run Ollama locally."
        )

    # ── Checkpoint: load already-processed modules ────────────────────────
    already_done: set[str] = set()
    existing_rows: list[dict] = []
    if output_pairs_csv.exists():
        existing_df = pd.read_csv(output_pairs_csv)
        already_done = set(existing_df["module_code"].astype(str).str.strip().unique())
        existing_rows = existing_df.to_dict("records")
        print(f"[INFO] Resuming — {len(already_done)} modules already in output, "
              f"skipping those.", file=sys.stderr)

    remaining = df[~df["module_code"].astype(str).str.strip().isin(already_done)]
    print(f"[INFO] Extracting skills from {len(remaining)} modules using providers: {providers}",
          file=sys.stderr)

    all_rows: list[dict] = list(existing_rows)
    n = len(remaining)

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        code  = str(row["module_code"]).strip()
        title = str(row[title_col]).strip() if title_col else ""
        desc  = str(row[desc_col]).strip()

        if not desc or desc.lower() in ("nan", "none", ""):
            print(f"[{i}/{n}] {code}: no description, skipping.", file=sys.stderr)
            continue

        print(f"[{i}/{n}] {code} ...", file=sys.stderr, end=" ")

        try:
            prompt   = _build_prompt(desc)
            raw_text = call_llm(_SYSTEM_PROMPT, prompt, providers=providers)
            parsed   = _parse_response(raw_text)
            rows     = _to_skill_rows(parsed, code, title)
            all_rows.extend(rows)
            print(f"{len(rows)} skills", file=sys.stderr)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            # Save progress before propagating fatal errors (e.g. all providers exhausted)
            if "All LLM providers exhausted" in str(exc):
                _save(all_rows, output_pairs_csv)
                print(f"[INFO] Progress saved ({i - 1}/{n} modules). "
                      "Re-run tomorrow to continue.", file=sys.stderr)
                raise

        # Save after every 50 modules so progress survives crashes
        if i % 50 == 0:
            _save(all_rows, output_pairs_csv)
            print(f"[INFO] Checkpoint saved ({i}/{n} modules done).", file=sys.stderr)

    _save(all_rows, output_pairs_csv)

    n_modules = pd.DataFrame(all_rows)["module_code"].nunique() if all_rows else 0
    print(
        f"\n[INFO] Extracted {len(all_rows)} skill-pairs across {n_modules} modules.",
        file=sys.stderr,
    )
    print(f"[INFO] Saved → {output_pairs_csv}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM few-shot skill extractor for NUS course descriptions.",
        epilog="""
Splitting work across groupmates (example — 4 people, 8615 modules total):
  Person A: --start-idx 0    --end-idx 2154  --output data/processed/skills_A.csv
  Person B: --start-idx 2154 --end-idx 4308  --output data/processed/skills_B.csv
  Person C: --start-idx 4308 --end-idx 6462  --output data/processed/skills_C.csv
  Person D: --start-idx 6462               --output data/processed/skills_D.csv

Merging after everyone finishes:
  python -m src.data_processing.extract_course_skills_llm --merge \\
      data/processed/skills_A.csv data/processed/skills_B.csv \\
      data/processed/skills_C.csv data/processed/skills_D.csv \\
      --output data/processed/module_skill_pairs.csv
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", default="data/processed/cleaned_modules.csv",
        help="Cleaned modules CSV.",
    )
    parser.add_argument(
        "--output", default="data/processed/module_skill_pairs.csv",
        help="Output skill pairs CSV.",
    )
    parser.add_argument(
        "--start-idx", type=int, default=0,
        help="First row index to process (0-based). For splitting work across groupmates.",
    )
    parser.add_argument(
        "--end-idx", type=int, default=None,
        help="Exclusive end row index. Omit to process to the last row.",
    )
    parser.add_argument(
        "--providers", nargs="+",
        choices=["cerebras", "sambanova", "groq", "gemini", "anthropic", "openai", "ollama"],
        default=None,
        help="LLM providers to try in order. Auto-detected from env vars if omitted.",
    )
    parser.add_argument(
        "--merge", nargs="+", metavar="CSV",
        help="Merge mode: combine partial CSVs from groupmates into --output.",
    )
    args = parser.parse_args()

    if args.merge:
        # Merge mode — combine partial output files
        dfs = [pd.read_csv(f) for f in args.merge]
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates(
            subset=["module_code", "skill_label"]
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.output, index=False)
        print(f"Merged {len(args.merge)} files → {len(merged)} rows → {args.output}")
    else:
        extract_all(
            input_csv=Path(args.input),
            output_pairs_csv=Path(args.output),
            providers=args.providers,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )
