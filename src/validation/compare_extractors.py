"""
compare_extractors.py
=====================
Skill extraction comparison for NUS course descriptions.

Primary evaluation used in the final project:
  1  SkillNer alone            -- initial baseline: raw NER, no taxonomy
  3  LLM few-shot              -- final model: prompt with skill definition + annotation examples

Legacy / optional variants:
  2  SkillNer + taxonomy       -- not used in final validation
  4  Hybrid: SkillNer -> LLM   -- not used in final validation: SkillNer provides candidates; LLM filters & expands

Recommended LLM backend — Ollama (free, local, no rate limits):
  1. Install: https://ollama.com
  2. Pull a model: ollama pull llama3.2
  3. Run (once): ollama serve
  Then pass: --llm-provider ollama
  Model defaults to llama3.2. Override with OLLAMA_MODEL env var, e.g.:
    export OLLAMA_MODEL=llama3.1:8b

Other providers:
  export CEREBRAS_API_KEY=...   (cloud.cerebras.ai, free tier: very generous)
  export GEMINI_API_KEY=...     (aistudio.google.com/apikey, free: 1M tokens/day)
  export GROQ_API_KEY=...       (console.groq.com, free tier: 100k tokens/day)
  export ANTHROPIC_API_KEY=...  (console.anthropic.com)
  export OPENAI_API_KEY=...     (platform.openai.com)

Usage:
    python -m src.validation.compare_extractors \\
        --courses path/to/cleaned_modules.csv \\
        --ground-truth validation/annotations.xlsx \\
        --methods 1 3 \\
        --llm-provider ollama \\
        --output validation/extractor_comparison_2_methods.csv

Ground truth Excel format:
    Sheet name: "courses_annotated"  (falls back to first sheet)
    Required columns: module_code, skill_label
    Optional columns: skill_type, knowledge_type, sentence_type
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT))

# Load .env so GROQ_API_KEY etc. are available without manual export
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExtractedSkill:
    skill_label: str
    skill_type: str = "unknown"      # "hard" | "soft" | "unknown"
    knowledge_type: str = "unknown"  # "applied" | "theoretical" | "unknown"


@dataclass
class MethodResult:
    module_code: str
    method_name: str
    skills: list[ExtractedSkill] = field(default_factory=list)
    error: str | None = None


# ---------------------------------------------------------------------------
# Normalization and fuzzy matching
# ---------------------------------------------------------------------------

_PUNC_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _normalize(skill: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, simple singularize.

    Singularize: remove trailing 's' only for words >= 5 characters, which
    avoids stripping meaningful endings like 'class' or 'pass'.
    """
    s = skill.lower().strip()
    s = _PUNC_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s).strip()
    words = [w[:-1] if (w.endswith("s") and len(w) >= 5) else w for w in s.split()]
    return " ".join(words)


def skills_match(pred: str, gt: str) -> bool:
    """Two-stage fuzzy match after normalization.

    Stage 1 — exact match.
    Stage 2 — substring containment (one skill name contains the other).

    Examples that match:
        "dynamic programming"  ↔  "dynamic programming algorithms"
        "graph algorithms"     ↔  "graph algorithm"
        "np-completeness"      ↔  "np completeness"
    """
    p = _normalize(pred)
    g = _normalize(gt)
    return p == g or p in g or g in p


def compute_metrics(predicted: list[str], ground_truth: list[str]) -> dict[str, Any]:
    """Precision / recall / F1 with fuzzy two-stage matching.

    Each ground truth skill is matched at most once (greedy left-to-right).
    Returns a dict with keys: precision, recall, f1, tp, fp, fn.
    """
    if not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    matched_gt: set[int] = set()
    tp = 0
    fp = 0

    for pred in predicted:
        found = False
        for i, gt in enumerate(ground_truth):
            if i not in matched_gt and skills_match(pred, gt):
                matched_gt.add(i)
                found = True
                break
        if found:
            tp += 1
        else:
            fp += 1

    fn = len(ground_truth) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn,
    }


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

# *** REPLACE THESE ENTRIES with your actual annotations when ready ***
# These are illustrative examples derived from annotation calibration
# sessions for CS3230 (Algorithms), CS4248 (NLP), and CS3244 (ML).
DUMMY_GROUND_TRUTH: dict[str, list[dict[str, str]]] = {
    "CS3230": [
        {"skill_label": "algorithm design",       "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "dynamic programming",    "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "greedy algorithms",      "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "graph algorithms",       "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "divide and conquer",     "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "computational complexity","skill_type": "hard", "knowledge_type": "theoretical"},
        {"skill_label": "network flow",           "skill_type": "hard", "knowledge_type": "theoretical"},
        {"skill_label": "np-completeness",        "skill_type": "hard", "knowledge_type": "theoretical"},
        {"skill_label": "linear programming",     "skill_type": "hard", "knowledge_type": "theoretical"},
    ],
    "CS4248": [
        {"skill_label": "natural language processing", "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "language modelling",          "skill_type": "hard", "knowledge_type": "theoretical"},
        {"skill_label": "text classification",         "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "machine translation",         "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "sentiment analysis",          "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "word embeddings",             "skill_type": "hard", "knowledge_type": "theoretical"},
    ],
    "CS3244": [
        {"skill_label": "machine learning",    "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "neural networks",     "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "regression",          "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "classification",      "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "model evaluation",    "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "feature engineering", "skill_type": "hard", "knowledge_type": "applied"},
        {"skill_label": "convex optimisation", "skill_type": "hard", "knowledge_type": "theoretical"},
    ],
}


def load_ground_truth(excel_path: str | None) -> dict[str, list[dict[str, str]]]:
    """Load ground truth from annotation Excel, or fall back to dummy.

    Expected Excel format:
        Sheet:   "annotations"  (falls back to first sheet if not found)
        Columns: module_code (required), skill_label (required),
                 skill_type (optional), knowledge_type (optional)

    Returns: {module_code: [{"skill_label": ..., "skill_type": ..., "knowledge_type": ...}]}
    """
    if excel_path is None:
        print(
            "[INFO] No --ground-truth provided. Using dummy (3 courses). "
            "Pass --ground-truth <path> when your annotation Excel is ready.",
            file=sys.stderr,
        )
        return DUMMY_GROUND_TRUTH

    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")

    try:
        df = pd.read_excel(path, sheet_name="courses_annotated")
    except Exception:
        df = pd.read_excel(path)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    required = {"module_code", "skill_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Ground truth Excel missing columns: {missing}. Found: {sorted(df.columns)}"
        )

    gt: dict[str, list[dict[str, str]]] = {}
    for _, row in df.iterrows():
        code = str(row["module_code"]).strip()
        if not code or code.lower() == "nan":
            continue
        gt.setdefault(code, []).append({
            "skill_label":    str(row["skill_label"]).strip(),
            "skill_type":     str(row.get("skill_type", "unknown")).strip(),
            "knowledge_type": str(row.get("knowledge_type", "unknown")).strip(),
        })

    total_skills = sum(len(v) for v in gt.values())
    print(f"[INFO] Ground truth: {len(gt)} courses, {total_skills} annotations.", file=sys.stderr)
    return gt


# ---------------------------------------------------------------------------
# SkillNer singleton (shared by methods 1 and 2)
# ---------------------------------------------------------------------------

def _load_skillner():
    if not hasattr(_load_skillner, "_extractor"):
        try:
            import spacy
            from spacy.matcher import PhraseMatcher
            from skillNer.general_params import SKILL_DB
            from skillNer.skill_extractor_class import SkillExtractor
        except ImportError as exc:
            raise RuntimeError(
                f"SkillNer not installed: {exc}\n"
                "  pip install skillner spacy\n"
                "  python -m spacy download en_core_web_lg"
            ) from exc
        try:
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            nlp = spacy.load("en_core_web_sm")
        _load_skillner._extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
        print("[INFO] SkillNer loaded.", file=sys.stderr)
    return _load_skillner._extractor


# ---------------------------------------------------------------------------
# Method 1 — SkillNer alone
# ---------------------------------------------------------------------------

def method1_skillner_only(description: str) -> list[ExtractedSkill]:
    """Raw SkillNer NER output.  No taxonomy, no post-filtering.

    Baseline to isolate SkillNer's native performance on course descriptions.
    """
    extractor = _load_skillner()
    try:
        annotation = extractor.annotate(description)
    except Exception:
        return []

    results = annotation.get("results", {})
    out: list[ExtractedSkill] = []
    seen: set[str] = set()

    for bucket in ("full_matches", "ngram_scored"):
        for item in results.get(bucket, []):
            raw = (
                item.get("doc_node_value")
                or item.get("skill_name")
                or item.get("text")
                or ""
            ).strip().lower()
            if not raw or len(raw) <= 2 or raw in seen:
                continue
            seen.add(raw)
            out.append(ExtractedSkill(skill_label=raw))

    return out


# ---------------------------------------------------------------------------
# Method 2 — SkillNer + taxonomy (existing pipeline)
# ---------------------------------------------------------------------------

def _load_pipeline_state():
    """Load the existing pipeline state once and cache it."""
    if not hasattr(_load_pipeline_state, "_state"):
        from src.data_processing.extract_course_skills import (
            build_skillner_extractor,
            load_taxonomy,
            _load_skillner_type_lookup,
            PipelineConfig,
        )
        extractor, _ = build_skillner_extractor()
        taxonomy      = load_taxonomy(None)
        id_to_type, name_to_type = _load_skillner_type_lookup()
        _load_pipeline_state._state = (extractor, taxonomy, id_to_type, name_to_type, PipelineConfig())
    return _load_pipeline_state._state


def method2_skillner_taxonomy(row: pd.Series) -> list[ExtractedSkill]:
    """Existing pipeline: SkillNer + taxonomy merge + zero-skill recovery."""
    from src.data_processing.extract_course_skills import (
        extract_sections,
        extract_skillner_candidates,
        extract_taxonomy_candidates,
        merge_candidates,
        apply_zero_skill_recovery,
    )
    extractor, taxonomy, id_to_type, name_to_type, config = _load_pipeline_state()

    sections  = extract_sections(row)
    text      = sections["description_for_extraction"]
    sentences = sections["sentences"]

    annotation  = extractor.annotate(text) if extractor else {"results": {}}
    sn_cands    = extract_skillner_candidates(annotation, taxonomy, sentences, id_to_type, name_to_type)
    tax_cands   = extract_taxonomy_candidates(text, taxonomy, sentences)
    final, _    = merge_candidates(sn_cands, tax_cands, taxonomy, sections, config)
    final       = apply_zero_skill_recovery(sections["title"], sections["description"], final, taxonomy)

    return [
        ExtractedSkill(
            skill_label=c["canonical_skill"],
            skill_type=c.get("skill_type", "unknown"),
        )
        for c in final
    ]


# ---------------------------------------------------------------------------
# LLM helpers (shared by methods 3 and 4)
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

# Few-shot examples from manual annotation (data/validation/annotations.xlsx, sheet: few-shot examples).
# Modules: ACC1701, BMI1102, EN2207, GE3219, CDE3505, DSA4211.
# CDE3505 is a deliberate zero-skill example — approach-descriptor language ("data-driven, innovative
# systems approach") does NOT count as explicitly taught skills.
# GE3219 illustrates "understanding of X" framing → X is the skill (framing verb stripped from label).
_FEW_SHOT_EXAMPLES: list[dict] = [
    {
        # ACC1701 — Business & Finance, level 1000
        # Shows: "covered from viewpoint of" → theoretical; "also covered" with action context → applied
        "description": (
            "an introduction to accounting from a user perspective. Financial reporting is covered "
            "from the viewpoint of an external investor. The focus is on how accounting can help "
            "investors make better decisions. Book-keeping and preparation of financial statements "
            "are also covered at an introductory level, as investors need to be aware of how the "
            "financial statements are derived."
        ),
        "skills": [
            {"skill_label": "accounting",                        "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "financial reporting",               "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "book-keeping",                      "skill_type": "hard", "knowledge_type": "applied"},
            {"skill_label": "preparation of financial statements","skill_type": "hard", "knowledge_type": "applied"},
        ],
    },
    {
        # BMI1102 — Life Sciences / Health Informatics, level 1000
        # Shows: "introduction to X" → theoretical; "encompasses" → applied when action-oriented
        "description": (
            "the biomedical informatics student to the health sciences. This encompasses an "
            "introduction to clinical practice, and an overview of the underlying biology and "
            "manifestations of selected disease states. Besides this an overview of the information "
            "gathering and reasoning processes used to detect, understand and treat diseases will be "
            "provided. to provide a functional background to non-clinicians who are new to the "
            "healthcare system."
        ),
        "skills": [
            {"skill_label": "clinical practice",           "skill_type": "hard", "knowledge_type": "applied"},
            {"skill_label": "biology of disease states",   "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "manifestations of disease states", "skill_type": "hard", "knowledge_type": "theoretical"},
        ],
    },
    {
        # EN2207 — Humanities, level 2000
        # Shows: topic listings → theoretical; "foster effective strategies for" → applied
        "description": (
            "an introduction to some of the key concepts and theoretical perspectives in gender, "
            "sexuality, and feminist studies. We will examine literature and theory as major sources "
            "of feminist knowledge production. on developing analytical skills for examining the "
            "politics of gender and feminism as expressed in the formal, structural, and thematic "
            "elements of both critical and creative texts. cultivating strong writing practices, it "
            "also aims to foster effective strategies for academic research and writing that "
            "demonstrate an informed awareness of the wider contextual debates and methodologies in "
            "feminist and literary studies."
        ),
        "skills": [
            {"skill_label": "feminist studies",  "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "gender studies",    "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "sexuality studies", "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "academic research", "skill_type": "hard", "knowledge_type": "applied"},
            {"skill_label": "academic writing",  "skill_type": "hard", "knowledge_type": "applied"},
        ],
    },
    {
        # GE3219 — Social Sciences / Geography, level 3000
        # Shows: "understanding of X" → strip framing, X is the skill (theoretical)
        "description": (
            "to provide students with an in-depth understanding of the social, political, and "
            "economic changes at various geographical scales with respect to globalisation. More "
            "specifically, on developing understandings of the complex forces driving globalisation "
            "and the related urban and regional changes and the relationship between globalisation "
            "and regionalisation. is not just for geography students, but for all students who are "
            "interested in the urban and regional changes in the Asia-Pacific with respect to "
            "globalisation and regionalisation and the driving forces of the changes."
        ),
        "skills": [
            {"skill_label": "forces driving globalisation",              "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "urban and regional changes",                "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "relationship between globalisation and regionalisation", "skill_type": "hard", "knowledge_type": "theoretical"},
        ],
    },
    {
        # CDE3505 — Engineering / Urban Planning, level 3000
        # Shows: approach-descriptor language ("data-driven, innovative, systems approach") does NOT
        # count as an explicitly taught skill — extract nothing.
        "description": (
            "The issues that cities face are constantly evolving, and the way in which we address "
            "these issues must evolve alongside them. Singapore's planning framework was developed "
            "in a context that differs from that of today where we now face novel, complex issues "
            "such as an ageing population, climate change, and sustainability. How will we modify "
            "our approach to planning to address these issues? students to how Singapore engages "
            "with urban challenges that address the Integrated Master Planning & Development and "
            "Dynamic Urban Governance systems within the Liveability Framework through a "
            "data-driven, innovative, systems approach."
        ),
        "skills": [],
    },
    {
        # DSA4211 — Data Science & Statistics, level 4000
        # Shows: long "Topics include X, Y, Z" listing → all theoretical; no action verbs → no applied
        "description": (
            "Dimensionality is an issue that can arise in many scientific fields such as medicine, "
            "genetics, business and finance, among others. The statistical properties of estimation "
            "and inference procedures must be carefully established when the number of variables is "
            "much larger than the number of observations. discuss several statistical methodologies "
            "useful for exploring voluminous data. They include principal component analysis, "
            "clustering and classification, tree-structured analysis, neural network, hidden Markov "
            "models, sliced inverse regression, multiple testing, sure independent screening (SIS) "
            "and penalized estimation for variable selection. Real data will be used for "
            "illustration of these methods. Some fundamental theory for high-dimensional learning."
        ),
        "skills": [
            {"skill_label": "principal component analysis", "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "clustering",                   "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "classification",               "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "tree-structured analysis",     "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "neural network",               "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "hidden markov models",         "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "sliced inverse regression",    "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "multiple testing",             "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "sure independent screening",   "skill_type": "hard", "knowledge_type": "theoretical"},
            {"skill_label": "penalized estimation",         "skill_type": "hard", "knowledge_type": "theoretical"},
        ],
    },
]


def _build_fewshot_prompt(description: str) -> str:
    parts = []
    for ex in _FEW_SHOT_EXAMPLES:
        parts.append(
            f"Course description:\n{ex['description']}\n\n"
            f"Extracted skills:\n{json.dumps(ex['skills'], indent=2)}"
        )
    parts.append(f"Course description:\n{description}\n\nExtracted skills:")
    return "\n\n---\n\n".join(parts)


def _call_llm_raw(system_prompt: str, user_message: str, provider: str) -> str:
    """Call the configured LLM provider and return raw text response.

    Retries up to 3 times on per-minute rate limit (429 TPM). Does NOT retry
    on per-day limit (429 TPD) — those require waiting for the daily reset.
    """
    import time

    if provider == "groq":
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                return resp.choices[0].message.content
            except Exception as exc:
                msg = str(exc)
                if "tokens per day" in msg.lower() or "tpd" in msg.lower():
                    raise  # daily limit — no point retrying
                if "429" in msg and attempt < 2:
                    wait = 60 * (attempt + 1)
                    print(f"[WARN] Groq rate limit (per-minute), waiting {wait}s ...", file=sys.stderr)
                    time.sleep(wait)
                else:
                    raise

    elif provider == "anthropic":
        import anthropic  # pip install anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text

    elif provider == "openai":
        from openai import OpenAI  # pip install openai
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    elif provider == "cerebras":
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["CEREBRAS_API_KEY"],
            base_url="https://api.cerebras.ai/v1",
        )
        resp = client.chat.completions.create(
            model="llama3.1-8b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            temperature=0.0,
            max_completion_tokens=512,
        )
        return resp.choices[0].message.content

    elif provider == "gemini":
        from google import genai  # pip install google-genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
                max_output_tokens=512,
            ),
        )
        return resp.text

    elif provider == "ollama":
        import urllib.request
        model = os.environ.get("OLLAMA_MODEL", "llama3.2")
        host  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{host}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        return data["message"]["content"]

    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose: gemini, ollama, groq, anthropic, openai")


def _parse_llm_json(text: str) -> list[dict]:
    """Extract a JSON array from LLM response text (strips markdown fences)."""
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


def _skills_from_llm_response(raw: list[dict]) -> list[ExtractedSkill]:
    return [
        ExtractedSkill(
            skill_label=str(item.get("skill_label", "")).strip().lower(),
            skill_type=str(item.get("skill_type", "unknown")).lower(),
            knowledge_type=str(item.get("knowledge_type", "unknown")).lower(),
        )
        for item in raw
        if str(item.get("skill_label", "")).strip()
    ]


def _detect_provider() -> str | None:
    # Check Ollama first — if it's running locally, prefer it (free, no limits)
    try:
        import urllib.request
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return "ollama"
    except Exception:
        pass
    for key, name in [
        ("CEREBRAS_API_KEY", "cerebras"),
        ("GEMINI_API_KEY", "gemini"),
        ("GROQ_API_KEY", "groq"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
    ]:
        if os.environ.get(key):
            return name
    return None


# ---------------------------------------------------------------------------
# Method 3 — LLM few-shot
# ---------------------------------------------------------------------------

_M3_SYSTEM = f"""You are a skill extraction assistant for NUS university course descriptions.

SKILL DEFINITION: {_SKILL_DEFINITION}

{_ANNOTATION_RULES}

OUTPUT: Return ONLY a valid JSON array. No explanation text. No markdown.
Example element: {{"skill_label": "dynamic programming", "skill_type": "hard", "knowledge_type": "applied"}}"""


def method3_llm_fewshot(description: str, provider: str) -> list[ExtractedSkill]:
    """LLM with skill definition + annotated few-shot examples."""
    user_msg = _build_fewshot_prompt(description)
    raw_text = _call_llm_raw(_M3_SYSTEM, user_msg, provider)
    return _skills_from_llm_response(_parse_llm_json(raw_text))


# ---------------------------------------------------------------------------
# Method 4 — Hybrid: SkillNer candidates → LLM filter + expand
# ---------------------------------------------------------------------------

_M4_SYSTEM = f"""You are a skill extraction quality controller for NUS course descriptions.

You receive a course description and a list of candidate skills extracted by SkillNer (an automated NER tool).

Your tasks:
  A. FILTER  — Remove candidates that are NOT genuine skills by the definition below.
  B. EXPAND  — Add skills explicitly in the description that SkillNer missed.
  C. LABEL   — Assign skill_type and knowledge_type for all kept/added skills.

SKILL DEFINITION: {_SKILL_DEFINITION}

{_ANNOTATION_RULES}

OUTPUT: Return ONLY a valid JSON array. No explanation text. No markdown."""


def method4_hybrid(row: pd.Series, provider: str) -> list[ExtractedSkill]:
    """SkillNer extracts candidates; LLM filters false positives and recovers false negatives."""
    # Step 1: get SkillNer candidates
    sn_skills = method2_skillner_taxonomy(row)
    candidates = [s.skill_label for s in sn_skills]

    description = str(row.get("description_clean") or row.get("description") or "")

    # Build a few-shot example showing the filter+expand task.
    # Uses ACC1701 (_FEW_SHOT_EXAMPLES[0]): SkillNer would likely surface 'accounting' and
    # 'financial reporting' but miss 'book-keeping' and 'preparation of financial statements'.
    fewshot_ex = (
        f"Course description:\n"
        f"{_FEW_SHOT_EXAMPLES[0]['description']}\n\n"
        f"SkillNer candidates:\n"
        f"{json.dumps(['accounting', 'financial reporting', 'investor'], indent=2)}\n\n"
        f"Filtered and expanded skills:\n"
        f"{json.dumps(_FEW_SHOT_EXAMPLES[0]['skills'], indent=2)}"
    )

    user_msg = (
        f"{fewshot_ex}\n\n---\n\n"
        f"Course description:\n{description}\n\n"
        f"SkillNer candidates:\n{json.dumps(candidates, indent=2)}\n\n"
        f"Filtered and expanded skills:"
    )

    raw_text = _call_llm_raw(_M4_SYSTEM, user_msg, provider)
    return _skills_from_llm_response(_parse_llm_json(raw_text))


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_method(
    method_id: int,
    row: pd.Series,
    provider: str | None,
) -> MethodResult:
    """Run a single extractor on one course row. Returns MethodResult."""
    module_code = str(row.get("module code", "UNKNOWN")).strip()
    description = str(row.get("description_clean") or row.get("description") or "")
    method_names = {1: "skillner_only", 2: "skillner_taxonomy", 3: "llm_fewshot", 4: "hybrid"}
    method_name = method_names.get(method_id, f"method_{method_id}")

    try:
        if method_id == 1:
            skills = method1_skillner_only(description)
        elif method_id == 2:
            skills = method2_skillner_taxonomy(row)
        elif method_id == 3:
            if not provider:
                raise RuntimeError("No LLM API key found. Set GROQ_API_KEY (free at console.groq.com).")
            skills = method3_llm_fewshot(description, provider)
        elif method_id == 4:
            if not provider:
                raise RuntimeError("No LLM API key found. Set GROQ_API_KEY (free at console.groq.com).")
            skills = method4_hybrid(row, provider)
        else:
            raise ValueError(f"Unknown method id: {method_id}")

        return MethodResult(module_code=module_code, method_name=method_name, skills=skills)

    except Exception as exc:
        return MethodResult(
            module_code=module_code, method_name=method_name, error=str(exc)
        )


def evaluate(
    results: list[MethodResult],
    ground_truth: dict[str, list[dict[str, str]]],
) -> pd.DataFrame:
    """Compute precision/recall/F1 per course per method.

    Returns a DataFrame with columns:
        module_code, method, n_predicted, n_ground_truth,
        precision, recall, f1, tp, fp, fn,
        predicted_skills, missed_skills
    """
    rows = []
    for r in results:
        gt_entries = ground_truth.get(r.module_code, [])
        gt_labels  = [e["skill_label"] for e in gt_entries]
        pred_labels = [s.skill_label for s in r.skills]

        if r.error:
            metrics = {"precision": None, "recall": None, "f1": None,
                       "tp": None, "fp": None, "fn": None}
        else:
            metrics = compute_metrics(pred_labels, gt_labels)

        # Identify missed ground truth skills (false negatives)
        if not r.error and gt_labels:
            matched_gt: set[int] = set()
            for pred in pred_labels:
                for i, gt in enumerate(gt_labels):
                    if i not in matched_gt and skills_match(pred, gt):
                        matched_gt.add(i)
                        break
            missed = [gt_labels[i] for i in range(len(gt_labels)) if i not in matched_gt]
        else:
            missed = []

        rows.append({
            "module_code":     r.module_code,
            "method":          r.method_name,
            "n_predicted":     len(r.skills) if not r.error else None,
            "n_ground_truth":  len(gt_labels),
            "precision":       metrics["precision"],
            "recall":          metrics["recall"],
            "f1":              metrics["f1"],
            "tp":              metrics["tp"],
            "fp":              metrics["fp"],
            "fn":              metrics["fn"],
            "predicted_skills": "; ".join(s.skill_label for s in r.skills) if not r.error else f"ERROR: {r.error}",
            "missed_skills":   "; ".join(missed),
        })

    return pd.DataFrame(rows)


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-course metrics to per-method summary.

    Uses macro-averaged precision/recall/F1 across courses that have
    ground truth annotations.  Courses without ground truth are excluded.
    """
    df_with_gt = df[df["n_ground_truth"] > 0].copy()

    summary_rows = []
    for method, group in df_with_gt.groupby("method"):
        numeric = group[["precision", "recall", "f1", "tp", "fp", "fn"]].dropna()
        summary_rows.append({
            "method":            method,
            "courses_evaluated": len(numeric),
            "macro_precision":   round(numeric["precision"].mean(), 3),
            "macro_recall":      round(numeric["recall"].mean(), 3),
            "macro_f1":          round(numeric["f1"].mean(), 3),
            "total_tp":          int(numeric["tp"].sum()),
            "total_fp":          int(numeric["fp"].sum()),
            "total_fn":          int(numeric["fn"].sum()),
        })

    return pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)


def print_summary(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("EXTRACTOR COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Courses':>7} {'Precision':>10} {'Recall':>8} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 70)
    for _, r in summary.iterrows():
        print(
            f"{r['method']:<25} {r['courses_evaluated']:>7} "
            f"{r['macro_precision']:>10.3f} {r['macro_recall']:>8.3f} {r['macro_f1']:>6.3f} "
            f"{r['total_tp']:>5} {r['total_fp']:>5} {r['total_fn']:>5}"
        )
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare skill extraction methods on NUS course descriptions."
    )
    parser.add_argument(
        "--courses",
        required=True,
        help="Path to cleaned_modules.csv",
    )
    parser.add_argument(
        "--ground-truth",
        default=None,
        help="Path to annotation Excel (.xlsx). If omitted, uses dummy ground truth.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4],
        default=[1, 2],
        help="Which methods to run (default: 1 2). Methods 3 and 4 require an LLM API key.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["cerebras", "gemini", "ollama", "groq", "anthropic", "openai"],
        default=None,
        help="LLM provider (auto-detected: ollama if running, else env vars).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for per-course CSV output. Defaults to stdout-only.",
    )
    args = parser.parse_args()

    # Load data
    df_courses = pd.read_csv(args.courses)
    ground_truth = load_ground_truth(args.ground_truth)

    # Filter to courses that have ground truth (or all courses if no GT)
    course_codes = list(ground_truth.keys())
    if ground_truth is DUMMY_GROUND_TRUTH or not args.ground_truth:
        df_eval = df_courses[df_courses["module code"].isin(course_codes)].copy()
    else:
        df_eval = df_courses[df_courses["module code"].isin(course_codes)].copy()
        not_found = set(course_codes) - set(df_eval["module code"].tolist())
        if not_found:
            print(f"[WARN] {len(not_found)} annotated courses not found in CSV: {sorted(not_found)}", file=sys.stderr)

    if df_eval.empty:
        print("[ERROR] No matching courses found. Check that module codes in ground truth match 'module code' column in CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Evaluating on {len(df_eval)} courses.", file=sys.stderr)

    # Detect LLM provider
    provider = args.llm_provider or _detect_provider()
    if any(m in (3, 4) for m in args.methods) and not provider:
        print(
            "[WARN] Methods 3/4 require an LLM API key. "
            "Sign up free at console.groq.com and set GROQ_API_KEY.",
            file=sys.stderr,
        )

    # Run methods
    all_results: list[MethodResult] = []
    for method_id in args.methods:
        method_names = {1: "skillner_only", 2: "skillner_taxonomy", 3: "llm_fewshot", 4: "hybrid"}
        print(f"[INFO] Running method {method_id}: {method_names[method_id]} ...", file=sys.stderr)
        for _, row in df_eval.iterrows():
            result = run_method(method_id, row, provider)
            all_results.append(result)
            if result.error:
                print(f"  [WARN] {result.module_code}: {result.error}", file=sys.stderr)

    # Evaluate
    per_course_df = evaluate(all_results, ground_truth)
    summary_df    = aggregate_metrics(per_course_df)

    print_summary(summary_df)

    # Per-course breakdown
    print("\nPer-course breakdown:")
    print(per_course_df[["module_code", "method", "n_predicted", "n_ground_truth",
                          "precision", "recall", "f1", "tp", "fp", "fn"]].to_string(index=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_course_df.to_csv(out_path, index=False)

        summary_path = out_path.with_name(out_path.stem + "_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n[INFO] Saved per-course results → {out_path}", file=sys.stderr)
        print(f"[INFO] Saved summary           → {summary_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
