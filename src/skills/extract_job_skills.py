import argparse
import json
import multiprocessing as mp
import re
import unicodedata
from pathlib import Path

import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor


DEFAULT_INPUT = Path("jobs_processed.csv")
DEFAULT_OUTPUT = Path("jobs_processed_with_skills.csv")
TEXT_COLUMN = "clean_description"
FALLBACK_TEXT_COLUMN = "job_text"
WORKER_EXTRACTOR = None
WORKER_THRESHOLD = 0.5
SPACY_DISABLE = ("parser", "ner")
HEADER_TERMS = [
    "job description",
    "responsibilities",
    "requirements",
    "job scope",
    "duties",
]

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", flags=re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", flags=re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s()-]{6,}\d)\b")
MONEY_RE = re.compile(
    r"\b(?:sgd|s\$|\$)\s*\d[\d,]*(?:\.\d+)?(?:\s*[-–to]+\s*(?:sgd|s\$|\$)?\s*\d[\d,]*(?:\.\d+)?)?\+?\b",
    flags=re.IGNORECASE,
)
TIME_RANGE_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)\s*[-–to]+\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)\b",
    flags=re.IGNORECASE,
)
NOISE_LINE_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bea license\b",
        r"\bea personnel\b",
        r"\bworking hours?\b",
        r"\bshift [ab]\b",
        r"\brest day\b",
        r"\bsalary(?:\s*&\s*benefits)?\b",
        r"\bbasic salary\b",
        r"\bestimated monthly gross\b",
        r"\bcommission scheme\b",
        r"\bvariable bonus\b",
        r"\bannual leave\b",
        r"\bhow to apply\b",
        r"\bapply now\b",
        r"\bsend (?:your|resume|cv)\b",
        r"\bwhatsapp\b",
        r"\bcontact us\b",
        r"\blocation\s*:",
        r"\bresume\b",
        r"\bavailability\b",
        r"\bea licen[cs]e\b",
    )
]
SECTION_HEADER_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^\s*(?:⏰\s*)?working hours?(?:\s*&\s*shifts?)?\b",
        r"^\s*(?:⏰\s*)?shifts?\b",
        r"^\s*(?:💰\s*)?salary(?:\s*&\s*benefits)?\b",
    )
]
SECTION_END_HINTS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^\s*(?:🍽️?\s*)?key responsibilities\b",
        r"^\s*(?:💼\s*)?requirements?\b",
        r"^\s*(?:job|role)\s+descriptions?\b",
        r"^\s*(?:about us|overview|summary)\b",
        r"^\s*ready to\b",
    )
]
RECRUITER_LINE_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bresume\b",
        r"\bavailability\b",
        r"\bea licen[cs]e\b",
        r"\bea personnel\b",
        r"\bhrnet\b",
        r"\br\d{7,}\b",
    )
]
LINE_ONLY_NOISE_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"^\s*(?:📍\s*)?location\b",
        r"^\s*contract period\b",
        r"^\s*(?:📧\s*)?email\b",
        r"^\s*working days?\b",
        r"^\s*by sending us your personal data\b",
        r"^\s*you acknowledge that\b",
        r"^\s*if you wish to withdraw your consent\b",
        r"^\s*please feel free to contact us\b",
        r"^\s*privacy policy\b",
        r"^\s*terms of use\b",
    )
]
VALUE_ERROR_TOKEN_RE = re.compile(r"^'(.+)' is not in list$")
HEADER_TERMS_RE = re.compile(
    r"\b(?:job\s+)?(?:"
    + "|".join(re.escape(term) for term in HEADER_TERMS)
    + r")\b\s*[:.]?",
    flags=re.IGNORECASE,
)
INLINE_NOISE_REPLACEMENTS = [
    (re.compile(r"\bworking days?\s*:[^\n]*", flags=re.IGNORECASE), " "),
    (re.compile(r"\blocation\s*:[^\n]*", flags=re.IGNORECASE), " "),
    (re.compile(r"\bcontract period\s*:[^\n]*", flags=re.IGNORECASE), " "),
    (
        re.compile(r"\b[a-z][a-z\s]+\(eap no\.[^)]+\)\s*ea licen[cs]e no\.[^.\\n]*", flags=re.IGNORECASE),
        " ",
    ),
]


# -----------------------------
# Preprocessing
# -----------------------------

def strip_noise_lines(text: str) -> str:
    kept_lines = []
    skip_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            skip_section = False
            continue
        if any(pattern.search(line) for pattern in SECTION_HEADER_PATTERNS):
            skip_section = True
            continue
        if skip_section:
            if any(pattern.search(line) for pattern in SECTION_END_HINTS):
                skip_section = False
            else:
                continue
        if any(pattern.search(line) for pattern in LINE_ONLY_NOISE_PATTERNS):
            continue
        if any(pattern.search(line) for pattern in NOISE_LINE_PATTERNS):
            continue
        if any(pattern.search(line) for pattern in RECRUITER_LINE_PATTERNS):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def preprocess_for_skillner(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip()
    if not text or text == "#NAME?":
        return ""

    # Normalize unicode and preserve the content of angle-bracket snippets.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<([^>]+)>", r" \1 ", text)

    # Standardize punctuation and symbols that commonly break tokenization.
    replacements = {
        "’": "'",
        "“": '"',
        "”": '"',
        "→": " to ",
        "–": " - ",
        "—": " - ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    # Expand or rewrite high-frequency phrase patterns before SkillNer sees them.
    text = re.sub(r'[“"]?P[”"]s', "Ps", text)
    text = re.sub(r"\bwe've\b", "we have", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwe'll\b", "we will", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthey're\b", "they are", text, flags=re.IGNORECASE)
    text = re.sub(r"\bit's\b", "it is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwho's\b", "who is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwhat's\b", "what is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhow's\b", "how is", text, flags=re.IGNORECASE)
    text = re.sub(r"\bable to\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bids\b", " identifiers ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bp\s*(?:and|&)\s*ids\b", " process and instrumentation diagrams ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpiping\b", " piping systems ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bweds\b", "aligns", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwedding\b", "marriage", text, flags=re.IGNORECASE)
    for pattern, replacement in INLINE_NOISE_REPLACEMENTS:
        text = pattern.sub(replacement, text)

    # Remove emoji while preserving line boundaries for section-based stripping.
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    text = re.sub(r"[•●]", " ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Remove recruiter metadata and repeated non-skill boilerplate sections.
    text = strip_noise_lines(text)

    # Remove structured noise like contacts, URLs, salary, and time ranges.
    text = EMAIL_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = PHONE_RE.sub(" ", text)
    text = MONEY_RE.sub(" ", text)
    text = TIME_RANGE_RE.sub(" ", text)
    text = re.sub(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:weekday|weekend|public holiday|office hours)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:aircon|bbq|glow-up|jet-setter)\b", " ", text, flags=re.IGNORECASE)

    # Flatten bullets and remove decorative symbols that create bad spans.
    text = re.sub(r"(?<=\w)&(?=\w)", " and ", text)
    text = re.sub(r"(?<=\w)/(?=\w)", " or ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\n\s*[-*]\s*", ". ", text)
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\b[ivxlcdm]+\.\s*", " ", text, flags=re.IGNORECASE)
    text = HEADER_TERMS_RE.sub(" ", text)
    text = re.sub(r"(?:\s*\.\s*){2,}", ". ", text)
    text = re.sub(r"(^|\s)\.(?=\s|$)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# Extraction
# -----------------------------
def load_nlp(disable=SPACY_DISABLE):
    try:
        return spacy.load("en_core_web_lg", disable=list(disable)), "en_core_web_lg"
    except OSError:
        raise RuntimeError(
            "en_core_web_lg is not installed. Please install it with:\n"
            "python -m spacy download en_core_web_lg"
        )


def build_extractor(preferred_model=None):
    nlp, model_name = load_nlp(preferred_model)
    extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
    return extractor, model_name


def normalize_matches(annotation):
    skill_ids = []
    seen = set()
    for group in ("full_matches", "ngram_scored"):
        for match in annotation["results"][group]:
            skill_id = match["skill_id"]
            if skill_id in seen:
                continue
            seen.add(skill_id)
            skill_ids.append(skill_id)
    return skill_ids


def split_skill_types(skill_ids):
    skills = []
    hard_skills = []
    soft_skills = []
    certifications = []

    for skill_id in skill_ids:
        skill = SKILL_DB[skill_id]
        skill_name = skill["skill_name"]
        skill_type = skill["skill_type"]
        skills.append(skill_name)
        if skill_type == "Hard Skill":
            hard_skills.append(skill_name)
        elif skill_type == "Soft Skill":
            soft_skills.append(skill_name)
        elif skill_type == "Certification":
            certifications.append(skill_name)

    return skills, hard_skills, soft_skills, certifications


def harden_for_index_error(text: str) -> str:
    # Aggressively simplify acronym-heavy engineering text before retrying SkillNer.
    text = re.sub(r"\([^)]*[A-Z]{2,}[^)]*\)", " ", text)
    text = re.sub(r"\b[A-Z]{2,}\b", " ", text)
    text = re.sub(r"\b[A-Za-z]*\d+[A-Za-z\d]*\b", " ", text)
    text = re.sub(r"[\\/;:|]", " ", text)
    text = re.sub(r"[^\w\s.,()\-+]", " ", text)
    text = re.sub(r"\bids?\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[a-z]{1,2}\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def annotate_with_retries(extractor, text: str, threshold: float):
    attempt_text = text
    for _ in range(5):
        try:
            return extractor.annotate(attempt_text, tresh=threshold), attempt_text, ""
        except IndexError:
            # Retry on stricter text when SkillNer's internal token spans drift.
            stricter_text = harden_for_index_error(attempt_text)
            if stricter_text == attempt_text or not stricter_text:
                return None, attempt_text, "IndexError: list index out of range"
            attempt_text = stricter_text
        except ValueError as exc:
            # Remove the offending token that SkillNer fails to map during scoring.
            match = VALUE_ERROR_TOKEN_RE.match(str(exc))
            if not match:
                return None, attempt_text, f"{type(exc).__name__}: {exc}"
            bad_token_raw = match.group(1)
            replacements = [bad_token_raw]
            if bad_token_raw.lower() == "id":
                replacements.extend(["ids", "p&id", "p and id", "p and ids"])
            if bad_token_raw.lower() == "able":
                replacements.append("able to")
            if bad_token_raw.lower() == "piping":
                replacements.append("piping systems")

            stricter_text = attempt_text
            for token in replacements:
                stricter_text = re.sub(rf"\b{re.escape(token)}\b", " ", stricter_text, flags=re.IGNORECASE)
            stricter_text = re.sub(r"[^\x00-\x7F]+", " ", stricter_text)
            stricter_text = re.sub(r"\s+", " ", stricter_text).strip()
            if stricter_text == attempt_text or not stricter_text:
                return None, attempt_text, f"{type(exc).__name__}: {exc}"
            attempt_text = stricter_text
        except Exception as exc:
            return None, attempt_text, f"{type(exc).__name__}: {exc}"

    return None, attempt_text, "RetryLimitError: exceeded retries while extracting skills"


def extract_text_skills(extractor, text: str, threshold: float):
    cleaned_text = preprocess_for_skillner(text)
    if not cleaned_text:
        return cleaned_text, [], [], [], [], ""

    annotation, final_text, error = annotate_with_retries(extractor, cleaned_text, threshold)
    if error:
        return final_text, [], [], [], [], error

    skill_ids = normalize_matches(annotation)
    skills, hard_skills, soft_skills, certifications = split_skill_types(skill_ids)
    return final_text, skills, hard_skills, soft_skills, certifications, ""


def extract_row_skills(extractor, clean_description: str, job_text: str, threshold: float):
    cleaned_text, skills, hard_skills, soft_skills, certifications, error = extract_text_skills(
        extractor,
        clean_description,
        threshold,
    )

    if skills or error or not isinstance(job_text, str) or not job_text.strip():
        return cleaned_text, skills, hard_skills, soft_skills, certifications, error

    if job_text.strip() == (clean_description or "").strip():
        return cleaned_text, skills, hard_skills, soft_skills, certifications, error

    fallback_cleaned, fallback_skills, fallback_hard, fallback_soft, fallback_certs, fallback_error = extract_text_skills(
        extractor,
        job_text,
        threshold,
    )
    if fallback_skills:
        return fallback_cleaned, fallback_skills, fallback_hard, fallback_soft, fallback_certs, fallback_error

    return cleaned_text, skills, hard_skills, soft_skills, certifications, error


def init_worker(model_name: str | None, threshold: float):
    global WORKER_EXTRACTOR, WORKER_THRESHOLD
    WORKER_EXTRACTOR, _ = build_extractor(model_name)
    WORKER_THRESHOLD = threshold


def process_row(row):
    clean_description, job_text = row
    return extract_row_skills(WORKER_EXTRACTOR, clean_description, job_text, WORKER_THRESHOLD)


def build_result_frame(df: pd.DataFrame, cleaned_texts, extracted_rows) -> pd.DataFrame:
    skill_columns = pd.DataFrame(
        extracted_rows,
        columns=[
            "skills",
            "hard_skills",
            "soft_skills",
            "certifications",
            "extraction_error",
        ],
    )

    result = df.iloc[: len(cleaned_texts)].copy()
    result["skillner_clean_description"] = cleaned_texts
    result = pd.concat([result.reset_index(drop=True), skill_columns.reset_index(drop=True)], axis=1)
    result["skill_count"] = result["skills"].apply(len)

    for col in ("skills", "hard_skills", "soft_skills", "certifications"):
        result[col] = result[col].apply(json.dumps)

    return result


def write_checkpoint(df: pd.DataFrame, cleaned_texts, extracted_rows, output_path: Path):
    checkpoint_path = output_path.with_name(f"{output_path.stem}.checkpoint.csv")
    progress_path = output_path.with_name(f"{output_path.stem}.progress.json")
    checkpoint = build_result_frame(df, cleaned_texts, extracted_rows)
    checkpoint.to_csv(checkpoint_path, index=False)
    progress_path.write_text(
        json.dumps(
            {
                "processed_rows": len(cleaned_texts),
                "total_rows": len(df),
                "checkpoint_file": str(checkpoint_path),
            },
            indent=2,
        )
    )


def run(input_path: Path, output_path: Path, threshold: float, model_name: str | None, workers: int):
    df = pd.read_csv(input_path).fillna("")

    cleaned_texts = []
    extracted_rows = []
    total_rows = len(df)
    rows = list(df[[TEXT_COLUMN, FALLBACK_TEXT_COLUMN]].itertuples(index=False, name=None))

    if workers <= 1:
        extractor, resolved_model_name = build_extractor(model_name)
        for idx, row in enumerate(rows, start=1):
            cleaned_text, skills, hard_skills, soft_skills, certifications, error = extract_row_skills(
                extractor,
                row[0],
                row[1],
                threshold,
            )
            cleaned_texts.append(cleaned_text)
            extracted_rows.append((skills, hard_skills, soft_skills, certifications, error))
            if idx % 500 == 0 or idx == total_rows:
                write_checkpoint(df, cleaned_texts, extracted_rows, output_path)
                print(f"Processed {idx}/{total_rows} jobs", flush=True)
    else:
        resolved_model_name = model_name or "en_core_web_lg"
        with mp.Pool(processes=workers, initializer=init_worker, initargs=(model_name, threshold)) as pool:
            for idx, result in enumerate(pool.imap(process_row, rows, chunksize=100), start=1):
                cleaned_text, skills, hard_skills, soft_skills, certifications, error = result
                cleaned_texts.append(cleaned_text)
                extracted_rows.append((skills, hard_skills, soft_skills, certifications, error))
                if idx % 500 == 0 or idx == total_rows:
                    write_checkpoint(df, cleaned_texts, extracted_rows, output_path)
                    print(f"Processed {idx}/{total_rows} jobs", flush=True)

    result = build_result_frame(df, cleaned_texts, extracted_rows)
    result.to_csv(output_path, index=False)

    print(f"Processed {len(result)} jobs with SkillNer", flush=True)
    print(f"spaCy model: {resolved_model_name}", flush=True)
    print(f"Input: {input_path}", flush=True)
    print(f"Output: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Extract skills from jobs_processed.csv with SkillNer.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model", default="en_core_web_lg")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    run(args.input, args.output, args.threshold, args.model, args.workers)


if __name__ == "__main__":
    main()
