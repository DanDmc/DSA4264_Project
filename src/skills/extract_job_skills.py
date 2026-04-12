"""
Explode and label job skills using only the `skills_list` column.

Workflow:
1. Load the filtered jobs CSV
2. Explode `skills_list` into one row per job-skill pair
3. Deduplicate unique skills for labeling
4. Label each unique skill with `skill_type` and `knowledge_type`
5. Patch labels back onto the exploded job-skill rows
6. Save a single labeled output CSV
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

load_dotenv(PROJECT_ROOT / ".env")

data_root = os.getenv("DATA_ROOT")
if not data_root:
    raise RuntimeError("DATA_ROOT is not set in the environment or .env file.")

DATA_ROOT = Path(data_root)
DEFAULT_INPUT = DATA_ROOT / "processed/jobs/03_jobs_filtered.csv"
DEFAULT_OUTPUT = DATA_ROOT / "processed/jobs/job_skill_pairs.csv"
DEFAULT_LOG = DATA_ROOT / "processed/jobs" / "labelling_runs" / "extract_job_skills.log"
DEFAULT_LABEL_CACHE = DATA_ROOT / "processed/jobs" / "labelling_runs" / "extract_job_skills_cache.csv"
JSON_BLOCK_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)

DEFAULT_KEEP_COLUMNS = [
    "job_id",
    "title",
    "ssoc_code",
    "category",
    "ssoc_submajor_code",
    "ssoc_submajor_title",
]

sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_provider import available_providers, call_llm


SYSTEM_PROMPT = (
    "Classify each skill. Return only a JSON array. "
    'Each item must be exactly {"row_id": <int>, "skill_type": "hard"|"soft", '
    '"knowledge_type": "applied"|"theoretical"}. '
    "No markdown. No code fences. No prose. "
    "Use soft for interpersonal or behavioral skills. "
    "Use hard for technical, tool, certification, or domain skills. "
    "Use theoretical for concepts, principles, standards, regulations, or knowledge domains. "
    "Otherwise use applied."
)

FEW_SHOT_EXAMPLES: list[dict[str, object]] = [
    {
        "row": {"row_id": 1, "skill": "Microsoft Excel"},
        "label": {"row_id": 1, "skill_type": "hard", "knowledge_type": "applied"},
    },
    {
        "row": {"row_id": 2, "skill": "Interpersonal Skills"},
        "label": {"row_id": 2, "skill_type": "soft", "knowledge_type": "applied"},
    },
    {
        "row": {"row_id": 3, "skill": "Accounting Standards"},
        "label": {"row_id": 3, "skill_type": "hard", "knowledge_type": "theoretical"},
    },
]

SOFT_KEYWORDS = {
    "adaptability", "attention to detail", "collaboration", "communication",
    "customer satisfaction", "customer service", "discipline", "empathy",
    "interpersonal skills", "leadership", "listening", "mentoring",
    "motivation", "negotiation", "people management", "presentation",
    "problem solving", "team player", "teamwork", "time management",
    "work independently",
}

HARD_KEYWORDS = {
    "accounting", "administration", "audit", "cctv", "civil engineering",
    "construction", "databases", "electrical", "facebook", "food safety",
    "hardware", "housekeeping", "inventory", "marketing", "microsoft excel",
    "microsoft office", "music education", "music theory", "networking",
    "opera", "piano", "public relations", "quality assurance", "sales",
    "scheduling", "software", "spreadsheets", "surveying",
    "technical support", "teaching", "training", "travel arrangements",
    "troubleshooting", "windows", "word processing",
}

THEORETICAL_KEYWORDS = {
    "accounting standards", "ethics", "law", "management science",
    "mathematics", "methodology", "music theory", "physics", "policies",
    "policy", "principles", "regulations", "requirements", "science",
    "standards", "statistics", "theory",
}

APPLIED_KEYWORDS = {
    "administration", "audit", "communication", "construction", "customer service",
    "event planning", "housekeeping", "inventory", "leadership", "marketing",
    "networking", "planning", "presentation", "public relations", "scheduling",
    "teaching", "technical support", "training", "travel arrangements",
    "troubleshooting", "word processing",
}


class _TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _parse_skills_list_cell(value: object) -> list[str]:
    text = str(value).strip()
    if not text:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        skill = part.strip()
        skill_key = _normalize_text(skill)
        if not skill or not skill_key or skill_key in seen:
            continue
        seen.add(skill_key)
        result.append(skill)
    return result


def _explode_skills_list_only(df: pd.DataFrame, keep_columns: list[str]) -> pd.DataFrame:
    if "job_id" not in df.columns or "skills_list" not in df.columns:
        raise ValueError("Input must contain 'job_id' and 'skills_list' columns.")

    available_keep_columns = [col for col in keep_columns if col in df.columns]
    rows: list[dict[str, object]] = []

    for row in df.itertuples(index=False):
        skills = _parse_skills_list_cell(getattr(row, "skills_list"))
        if not skills:
            continue

        base = {col: getattr(row, col) for col in available_keep_columns}
        for skill in skills:
            rows.append(
                {
                    **base,
                    "skill": skill,
                    "skill_type": "",
                    "knowledge_type": "",
                }
            )

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(
            columns=[*available_keep_columns, "skill", "skill_type", "knowledge_type"]
        )
    return result.sort_values(["job_id", "skill"]).reset_index(drop=True)


def _build_unique_skills_df(exploded_df: pd.DataFrame) -> pd.DataFrame:
    work = exploded_df[["skill"]].copy()
    work["skill"] = work["skill"].fillna("").astype(str).str.strip()
    work = work[work["skill"] != ""].copy()
    work["skill_key"] = work["skill"].map(_normalize_text)
    work = (
        work.sort_values(["skill_key", "skill"])
        .drop_duplicates(subset=["skill_key"], keep="first")
        .reset_index(drop=True)
    )
    work["skill_type"] = ""
    work["knowledge_type"] = ""
    return work


def _row_payload(row: pd.Series, row_id: int) -> dict[str, str | int]:
    return {"row_id": row_id, "skill": str(row.get("skill", "")).strip()}


def _build_prompt(rows: list[pd.Series]) -> str:
    parts: list[str] = ["Examples:"]
    for example in FEW_SHOT_EXAMPLES:
        parts.append(
            f"{json.dumps(example['row'], ensure_ascii=True, separators=(',', ':'))} -> "
            f"{json.dumps(example['label'], ensure_ascii=True, separators=(',', ':'))}"
        )
    payload = [_row_payload(row, row_id=i + 1) for i, row in enumerate(rows)]
    parts.append("Return labels for this input:")
    parts.append(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
    return "\n".join(parts)


def _parse_response(text: str) -> dict[int, dict[str, str]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates: list[str] = [cleaned]
    match = JSON_BLOCK_RE.search(cleaned)
    if match:
        candidates.append(match.group())

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(cleaned[start:end + 1])

    data = None
    for candidate in candidates:
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None or not isinstance(data, list):
        return {}

    parsed: dict[int, dict[str, str]] = {}
    valid_skill_types = {"hard", "soft"}
    valid_knowledge_types = {"theoretical", "applied"}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            row_id = int(item.get("row_id"))
        except (TypeError, ValueError):
            continue
        skill_type = str(item.get("skill_type", "")).strip().lower()
        knowledge_type = str(item.get("knowledge_type", "")).strip().lower()
        if skill_type not in valid_skill_types or knowledge_type not in valid_knowledge_types:
            continue
        parsed[row_id] = {"skill_type": skill_type, "knowledge_type": knowledge_type}
    return parsed


def _heuristic_skill_type(skill: object) -> str:
    text = _normalize_text(skill)
    if not text:
        return ""
    if text in SOFT_KEYWORDS:
        return "soft"
    if text in HARD_KEYWORDS:
        return "hard"
    if any(
        keyword in text
        for keyword in (
            "interpersonal",
            "communication skill",
            "customer service",
            "team player",
            "teamwork",
            "leadership",
            "work independently",
        )
    ):
        return "soft"
    if any(
        keyword in text
        for keyword in (
            "microsoft excel",
            "microsoft office",
            "software",
            "engineering",
            "technical support",
            "troubleshooting",
            "food safety",
            "quality assurance",
            "civil engineering",
        )
    ):
        return "hard"
    return ""


def _heuristic_knowledge_type(skill: object, skill_type: str) -> str:
    text = _normalize_text(skill)
    if not text:
        return ""
    if text in THEORETICAL_KEYWORDS:
        return "theoretical"
    if text in APPLIED_KEYWORDS:
        return "applied"
    if any(
        keyword in text
        for keyword in (
            " theory",
            "principle",
            "standard",
            "regulation",
            "policy",
            "science",
            "law",
            "methodology",
            "statistics",
        )
    ):
        return "theoretical"
    if skill_type == "soft":
        return "applied"
    if any(
        keyword in text
        for keyword in (
            "planning",
            "support",
            "teaching",
            "training",
            "customer service",
            "troubleshooting",
            "administration",
            "housekeeping",
        )
    ):
        return "applied"
    return ""


def _apply_heuristics(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = df.copy()
    assigned = 0
    for idx, row in result.iterrows():
        skill_type = str(row.get("skill_type", "")).strip().lower()
        knowledge_type = str(row.get("knowledge_type", "")).strip().lower()

        if not skill_type:
            skill_type = _heuristic_skill_type(row.get("skill", ""))
            if skill_type:
                result.at[idx, "skill_type"] = skill_type
                assigned += 1

        if not knowledge_type:
            knowledge_type = _heuristic_knowledge_type(row.get("skill", ""), skill_type)
            if knowledge_type:
                result.at[idx, "knowledge_type"] = knowledge_type
                assigned += 1

    return result, assigned


def _save_label_cache(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_columns = ["skill", "skill_type", "knowledge_type"]
    extra_columns = [c for c in df.columns if c not in ordered_columns and c != "skill_key"]
    df.loc[:, [*ordered_columns, *extra_columns]].to_csv(output_path, index=False)


def _patch_labels(exploded_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    result = exploded_df.copy()
    result["skill_key"] = result["skill"].map(_normalize_text)
    label_map = (
        labels_df[["skill_key", "skill_type", "knowledge_type"]]
        .drop_duplicates(subset=["skill_key"], keep="last")
        .set_index("skill_key")
        .to_dict("index")
    )

    for idx, row in result.iterrows():
        mapped = label_map.get(row["skill_key"])
        if not mapped:
            continue
        result.at[idx, "skill_type"] = mapped["skill_type"]
        result.at[idx, "knowledge_type"] = mapped["knowledge_type"]

    return result.drop(columns=["skill_key"], errors="ignore")


def _load_exploded_input(input_csv: Path, mode: str, keep_columns: list[str]) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(input_csv).fillna(""))
    normalized_keep_columns = [c.strip().lower().replace(" ", "_") for c in keep_columns]

    if mode == "explode_and_label":
        return _explode_skills_list_only(df, normalized_keep_columns)

    required_cols = {"skill"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Input must contain {sorted(missing)} when mode=label_only. "
            "Use an already exploded job-skill CSV."
        )

    exploded_df = df.copy()
    if "skill_type" not in exploded_df.columns:
        exploded_df["skill_type"] = ""
    if "knowledge_type" not in exploded_df.columns:
        exploded_df["knowledge_type"] = ""
    return exploded_df


def extract_and_label_job_skills(
    input_csv: Path,
    output_csv: Path,
    keep_columns: list[str],
    mode: str,
    providers: list[str] | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
    batch_size: int = 20,
    label_cache_csv: Path | None = None,
) -> None:
    exploded_df = _load_exploded_input(input_csv, mode, keep_columns)
    unique_skills_df = _build_unique_skills_df(exploded_df)
    total = len(unique_skills_df)
    unique_skills_df = unique_skills_df.iloc[start_idx:end_idx].copy()

    if start_idx or end_idx is not None:
        print(
            f"[INFO] Processing rows {start_idx}–{end_idx or total} ({len(unique_skills_df)} of {total} unique skills).",
            file=sys.stderr,
        )

    if providers is None:
        providers = available_providers()
    if not providers:
        raise RuntimeError("No LLM providers available. Set provider API keys in the project env or run Ollama locally.")

    cache_path = label_cache_csv or DEFAULT_LABEL_CACHE
    if cache_path.exists():
        cache_df = _normalize_columns(pd.read_csv(cache_path))
        if {"skill", "skill_type", "knowledge_type"}.issubset(cache_df.columns):
            cache_df["skill_key"] = cache_df["skill"].map(_normalize_text)
            complete_cache_df = cache_df[
                cache_df["skill_type"].fillna("").astype(str).str.strip().ne("")
                & cache_df["knowledge_type"].fillna("").astype(str).str.strip().ne("")
            ].copy()
            existing_map = (
                complete_cache_df[["skill_key", "skill_type", "knowledge_type"]]
                .drop_duplicates(subset=["skill_key"], keep="last")
                .set_index("skill_key")
                .to_dict("index")
            )
            for idx, row in unique_skills_df.iterrows():
                cached = existing_map.get(row["skill_key"])
                if cached:
                    unique_skills_df.at[idx, "skill_type"] = cached["skill_type"]
                    unique_skills_df.at[idx, "knowledge_type"] = cached["knowledge_type"]
            print(
                f"[INFO] Resuming — {len(complete_cache_df)} unique skills already fully labeled in cache.",
                file=sys.stderr,
            )

    unique_skills_df, heuristic_count = _apply_heuristics(unique_skills_df)
    if heuristic_count:
        print(f"[INFO] Heuristics assigned {heuristic_count} labels before LLM calls.", file=sys.stderr)

    pending_mask = (
        unique_skills_df["skill_type"].astype(str).str.strip().eq("")
        | unique_skills_df["knowledge_type"].astype(str).str.strip().eq("")
    )
    pending_indices = unique_skills_df.index[pending_mask].tolist()
    print(
        f"[INFO] Labeling {len(pending_indices)} unique skills using providers: {providers} "
        f"(batch_size={batch_size})",
        file=sys.stderr,
    )

    done = 0
    for batch_start in range(0, len(pending_indices), batch_size):
        batch_indices = pending_indices[batch_start:batch_start + batch_size]
        batch_rows = [unique_skills_df.loc[idx] for idx in batch_indices]
        preview = ", ".join(str(unique_skills_df.loc[idx, "skill"]).strip() for idx in batch_indices[:3])
        print(
            f"[{done + 1}-{done + len(batch_indices)}/{len(pending_indices)}] batch of "
            f"{len(batch_indices)} skills ({preview}) ...",
            end=" ",
            file=sys.stderr,
        )
        try:
            raw_text = call_llm(SYSTEM_PROMPT, _build_prompt(batch_rows), providers=providers)
            labels = _parse_response(raw_text)
            if not labels:
                print(
                    "[WARN] Unparseable LLM response preview: "
                    f"{raw_text[:500].replace(chr(10), ' ')}",
                    file=sys.stderr,
                )
            labeled_count = 0
            for row_id, idx in enumerate(batch_indices, start=1):
                label = labels.get(row_id)
                if not label:
                    continue
                unique_skills_df.at[idx, "skill_type"] = label["skill_type"]
                unique_skills_df.at[idx, "knowledge_type"] = label["knowledge_type"]
                labeled_count += 1
            print(f"{labeled_count}/{len(batch_indices)} parsed", file=sys.stderr)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            if "All LLM providers exhausted" in str(exc):
                partial = unique_skills_df[
                    unique_skills_df["skill_type"].astype(str).str.strip().ne("")
                    | unique_skills_df["knowledge_type"].astype(str).str.strip().ne("")
                ].copy()
                _save_label_cache(partial.drop(columns=["skill_key"], errors="ignore"), cache_path)
                labeled_output = _patch_labels(exploded_df, partial)
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                labeled_output.to_csv(output_csv, index=False)
                print("[INFO] Progress saved before exiting.", file=sys.stderr)
                raise

        done += len(batch_indices)
        if done % max(100, batch_size * 5) == 0:
            partial = unique_skills_df[
                unique_skills_df["skill_type"].astype(str).str.strip().ne("")
                | unique_skills_df["knowledge_type"].astype(str).str.strip().ne("")
            ].copy()
            _save_label_cache(partial.drop(columns=["skill_key"], errors="ignore"), cache_path)
            labeled_output = _patch_labels(exploded_df, partial)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            labeled_output.to_csv(output_csv, index=False)
            print(
                f"[INFO] Checkpoint saved ({done}/{len(pending_indices)} unique skills done).",
                file=sys.stderr,
            )

    result = unique_skills_df.copy()
    _save_label_cache(result.drop(columns=["skill_key"], errors="ignore"), cache_path)
    labeled_output = _patch_labels(exploded_df, result)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    labeled_output.to_csv(output_csv, index=False)
    print(f"[INFO] Saved {len(labeled_output)} labeled job-skill rows to {output_csv}.", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explode skills_list-only job skills and label them with skill_type and knowledge_type."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input jobs CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output labeled job-skill CSV path.")
    parser.add_argument(
        "--mode",
        choices=["explode_and_label", "label_only"],
        default="explode_and_label",
        help="Use explode_and_label for raw jobs input, or label_only for an already exploded job-skill CSV.",
    )
    parser.add_argument(
        "--keep-cols",
        nargs="+",
        default=DEFAULT_KEEP_COLUMNS,
        help="Metadata columns to carry into the exploded output if present.",
    )
    parser.add_argument("--start-idx", type=int, default=0, help="First unique-skill index to process (0-based).")
    parser.add_argument("--end-idx", type=int, default=None, help="Exclusive end unique-skill index.")
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=["cerebras", "sambanova", "groq", "gemini", "anthropic", "openai", "ollama"],
        default=None,
        help="LLM providers to try in order. Auto-detected if omitted.",
    )
    parser.add_argument("--batch-size", type=int, default=20, help="Number of unique skills per LLM call.")
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG, help="Run log path.")
    parser.add_argument("--label-cache", type=Path, default=DEFAULT_LABEL_CACHE, help="CSV cache of labeled unique skills.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", encoding="utf-8") as log_handle:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeStream(sys.stdout, log_handle)
        sys.stderr = _TeeStream(sys.stderr, log_handle)
        try:
            print(
                f"[INFO] Starting run: input={args.input} output={args.output} mode={args.mode} "
                f"batch_size={args.batch_size} providers={args.providers or 'auto'}",
                file=sys.stderr,
            )
            extract_and_label_job_skills(
                input_csv=args.input,
                output_csv=args.output,
                keep_columns=args.keep_cols,
                mode=args.mode,
                providers=args.providers,
                start_idx=args.start_idx,
                end_idx=args.end_idx,
                batch_size=args.batch_size,
                label_cache_csv=args.label_cache,
            )
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
