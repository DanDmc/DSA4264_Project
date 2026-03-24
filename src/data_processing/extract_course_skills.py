import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    # Keep normalization behavior consistent across the project.
    from src.data_processing.postprocess_skills import canonicalize, is_valid
except Exception:
    def canonicalize(skill: str) -> str:
        return " ".join(str(skill).strip().lower().split())

    def is_valid(skill: str) -> bool:
        s = str(skill).strip().lower()
        if len(s) <= 2:
            return False
        if s.replace(" ", "").isdigit():
            return False
        return True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    min_confidence: float = 0.26
    debug: bool = False
    diagnostics_out: str | None = "data/processed/extraction_run_diagnostics.json"


ACTION_VERBS = {
    "apply", "analyze", "analyse", "implement", "design", "build", "evaluate",
    "develop", "model", "optimize", "optimise", "solve", "deploy", "create",
}

GENERIC_SKILL_PENALTY = {
    "research", "innovation", "interactive", "translated", "experimental",
    "statistical method", "methodology",
}

SOFT_SKILL_DEFAULTS = {
    "communication", "leadership", "project management", "teamwork",
    "problem solving", "critical thinking", "presentation", "writing",
    "academic writing",
}

LOCAL_JUNK_STOPLIST = {
    "communicate", "investigate", "furnish", "illustrated", "stage", "track",
    "lecturers", "researchers", "sampling", "program generate", "interactive",
    "translated", "related industries", "healthcare professionals",
    "msc", "investigator",
    "social", "economic", "case", "perspective", "lectures", "readings",
    "targeted", "debates", "conceptual", "interaction", "integration",
    "read", "reading", "survey", "scale", "innovative", "managing",
    "governing", "integrated", "act", "professional", "economy",
}

GENERIC_SINGLE_TOKEN_DROP = {
    "social", "economic", "professional", "urban", "creative",
    "innovative", "conceptual", "interaction", "integration",
    "perspective", "debates", "survey", "scale", "case",
}

# Keep a tiny allowlist for short token-like skill mentions.
SHORT_TOKEN_ALLOWLIST = {"r", "c+", "c++"}

ZERO_SKILL_RECOVERY_RULES: list[tuple[str, list[str]]] = [
    (r"thesis|dissertation|capstone|honours project|master.?s project", ["research methods", "academic writing"]),
    (r"\blaw\b|legal|jurisdiction|arbitration|litigation|evidence", ["legal research", "writing"]),
    (r"architecture|architectural|urban|heritage|conservation|landscape", ["design thinking", "critical thinking"]),
    (r"history|philosophy|politics|literature|culture|sociology|anthropology", ["critical thinking", "writing"]),
    (r"music|performance|recital|audition|ensemble|theatre|film", ["presentation", "teamwork"]),
]


# Canonical fallback taxonomy with minimal metadata.
FALLBACK_SKILL_TAXONOMY = {
    "python": ["python"],
    "r programming": ["r programming", "r language", "(r)", " r "],
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
    # Added based on observed misses in recent extraction output.
    "programming": ["programming", "coding"],
    "analytics": ["analytics", "business analytics"],
    "climate change analysis": ["climate change", "climate risk"],
    "policy analysis": ["policy analysis", "impact evaluation", "public policy"],
    "operations management": ["operations", "operations management"],
    "investment analysis": ["investment", "investment analysis"],
    "legal research": ["legal research", "legal analysis", "legal system"],
    "legal writing": ["legal writing"],
    "international law": ["international law"],
    "civil law": ["civil law"],
    "arbitration": ["arbitration"],
    "intellectual property law": ["intellectual property", "intellectual property law"],
    "human rights law": ["human rights", "human rights law"],
    "sociological analysis": ["sociology", "social science"],
    "psychological analysis": ["psychology", "psychological"],
    "sustainability assessment": ["sustainability", "sustainable development"],
    "technical drawing": ["drawing", "technical drawing"],
    "landscape design": ["landscape", "landscape design"],
}


# ---------------------------------------------------------------------------
# Text preprocessing (Stage 1)
# ---------------------------------------------------------------------------

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def clean_text(text: object) -> str:
    if text is None or pd.isna(text):
        return ""
    s = html.unescape(str(text))
    s = HTML_TAG_RE.sub(" ", s)
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u00a0", " ")
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if not parts:
        return [text]
    return parts


def extract_sections(row: pd.Series) -> dict[str, Any]:
    title = clean_text(row.get("title", ""))
    description = clean_text(row.get("description", ""))

    outcomes_candidates = [
        row.get("learning_outcomes", ""),
        row.get("learning outcomes", ""),
        row.get("outcomes", ""),
        row.get("intended learning outcomes", ""),
    ]
    learning_outcomes = clean_text(next((x for x in outcomes_candidates if clean_text(x)), ""))

    description_for_extraction = clean_text(row.get("description_clean", "")) or description

    sentence_objects: list[dict[str, str]] = []
    for sent in split_sentences(description_for_extraction):
        sentence_objects.append({"section": "description", "text": sent})
    for sent in split_sentences(learning_outcomes):
        sentence_objects.append({"section": "learning_outcomes", "text": sent})

    return {
        "title": title,
        "description": description,
        "description_for_extraction": description_for_extraction,
        "learning_outcomes": learning_outcomes,
        "sentences": sentence_objects,
    }


# ---------------------------------------------------------------------------
# Taxonomy (Stage 3 grounding)
# ---------------------------------------------------------------------------


def _default_type_for_skill(skill: str) -> str:
    return "soft" if skill in SOFT_SKILL_DEFAULTS else "hard"


def build_default_taxonomy_entries() -> list[dict[str, Any]]:
    entries = []
    for canonical, synonyms in FALLBACK_SKILL_TAXONOMY.items():
        entries.append(
            {
                "canonical_skill": canonical,
                "synonyms": sorted({canonical, *synonyms}),
                "parent_category": "transferable" if canonical in SOFT_SKILL_DEFAULTS else "technical",
                "skill_type": _default_type_for_skill(canonical),
            }
        )
    return entries


def _load_user_taxonomy(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows: list[dict[str, Any]] = []
        if isinstance(data, dict):
            for canonical, value in data.items():
                if isinstance(value, dict):
                    rows.append(
                        {
                            "canonical_skill": canonical,
                            "synonyms": value.get("synonyms", [canonical]),
                            "parent_category": value.get("parent_category"),
                            "skill_type": value.get("skill_type"),
                        }
                    )
                elif isinstance(value, list):
                    rows.append(
                        {
                            "canonical_skill": canonical,
                            "synonyms": value,
                            "parent_category": None,
                            "skill_type": None,
                        }
                    )
        elif isinstance(data, list):
            for item in data:
                rows.append(
                    {
                        "canonical_skill": item.get("canonical_skill") or item.get("skill") or item.get("canonical"),
                        "synonyms": item.get("synonyms", []),
                        "parent_category": item.get("parent_category"),
                        "skill_type": item.get("skill_type"),
                    }
                )
        return rows

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        rows: list[dict[str, Any]] = []
        for _, r in df.iterrows():
            canonical = r.get("canonical_skill") or r.get("skill") or r.get("canonical")
            syn_raw = str(r.get("synonyms", "")).strip()
            synonyms = [s.strip() for s in syn_raw.split("|") if s.strip()] if syn_raw else []
            rows.append(
                {
                    "canonical_skill": canonical,
                    "synonyms": synonyms,
                    "parent_category": r.get("parent_category"),
                    "skill_type": r.get("skill_type"),
                }
            )
        return rows

    raise ValueError("Taxonomy file must be .json or .csv")


class Taxonomy:
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries_by_skill: dict[str, dict[str, Any]] = {}
        self.alias_to_canonical: dict[str, str] = {}
        self.alias_patterns: list[tuple[str, re.Pattern[str]]] = []

        for entry in entries:
            canonical_raw = entry.get("canonical_skill")
            if not canonical_raw:
                continue
            canonical = canonicalize(str(canonical_raw))
            if not canonical:
                continue

            existing = self.entries_by_skill.get(canonical, {})
            merged_synonyms = set(existing.get("synonyms", []))
            merged_synonyms.update(entry.get("synonyms", []) or [])
            merged_synonyms.add(canonical)

            skill_type = entry.get("skill_type") or existing.get("skill_type") or _default_type_for_skill(canonical)
            parent = entry.get("parent_category") or existing.get("parent_category")

            self.entries_by_skill[canonical] = {
                "canonical_skill": canonical,
                "synonyms": sorted({canonicalize(s) for s in merged_synonyms if canonicalize(s)}),
                "parent_category": parent,
                "skill_type": str(skill_type).lower() if skill_type else "unknown",
            }

        for canonical, meta in self.entries_by_skill.items():
            for alias in meta["synonyms"]:
                self.alias_to_canonical[alias] = canonical

        # Regex aliases for explicit fallback matching.
        for alias, canonical in self.alias_to_canonical.items():
            pattern = re.compile(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", re.IGNORECASE)
            self.alias_patterns.append((canonical, pattern))

    def normalize(self, skill_text: str) -> str:
        norm = canonicalize(skill_text)
        return self.alias_to_canonical.get(norm, norm)

    def get_meta(self, canonical_skill: str) -> dict[str, Any]:
        meta = self.entries_by_skill.get(canonical_skill, {})
        return {
            "parent_category": meta.get("parent_category"),
            "skill_type": meta.get("skill_type", "unknown"),
        }

    @property
    def canonical_skills(self) -> list[str]:
        return list(self.entries_by_skill.keys())


def load_taxonomy(custom_path: str | None) -> Taxonomy:
    base_entries = build_default_taxonomy_entries()
    if custom_path:
        user_entries = _load_user_taxonomy(Path(custom_path))
        base_entries.extend(user_entries)
    return Taxonomy(base_entries)


# ---------------------------------------------------------------------------
# SkillNER (Stage 2)
# ---------------------------------------------------------------------------


def _skill_type_to_bucket(skill_type: str | None) -> str:
    st = (skill_type or "").strip().lower()
    if st == "soft skill":
        return "soft"
    if st in {"hard skill", "certification"}:
        return "hard"
    return "unknown"


def _load_skillner_type_lookup() -> tuple[dict[str, str], dict[str, str]]:
    id_to_type: dict[str, str] = {}
    name_to_type: dict[str, str] = {}

    skill_db = None
    local_db = Path("skill_db_relax_20.json")
    if local_db.exists():
        try:
            skill_db = json.loads(local_db.read_text(encoding="utf-8"))
        except Exception:
            skill_db = None

    if skill_db is None:
        try:
            from skillNer.general_params import SKILL_DB
            skill_db = SKILL_DB
        except Exception:
            skill_db = {}

    if not isinstance(skill_db, dict):
        return id_to_type, name_to_type

    for skill_id, payload in skill_db.items():
        if not isinstance(payload, dict):
            continue
        bucket = _skill_type_to_bucket(payload.get("skill_type"))
        if bucket != "unknown":
            id_to_type[str(skill_id)] = bucket

        names = [payload.get("skill_name", "")]
        names.extend(payload.get("high_surfce_forms", []) or [])
        names.extend(payload.get("low_surface_forms", []) or [])
        for name in names:
            norm = canonicalize(str(name))
            if norm:
                name_to_type[norm] = bucket

    return id_to_type, name_to_type


class KeywordSkillExtractor:
    def __init__(self, taxonomy: Taxonomy) -> None:
        self.taxonomy = taxonomy

    def annotate(self, text: str) -> dict:
        matches = []
        for canonical, pattern in self.taxonomy.alias_patterns:
            if pattern.search(text):
                matches.append({"doc_node_value": canonical})
        return {"results": {"full_matches": matches, "ngram_scored": []}}


def build_skillner_extractor() -> tuple[Any | None, dict[str, Any]]:
    status = {
        "enabled": True,
        "available": False,
        "backend": "skillner",
        "error": None,
    }
    try:
        import spacy
        from spacy.matcher import PhraseMatcher
        from requests.exceptions import RequestException
        from skillNer.general_params import SKILL_DB
        from skillNer.skill_extractor_class import SkillExtractor

        try:
            nlp = spacy.load("en_core_web_lg")
            status["spacy_model"] = "en_core_web_lg"
        except OSError:
            nlp = spacy.load("en_core_web_sm")
            status["spacy_model"] = "en_core_web_sm"

        print("Using extractor backend: SkillNER", file=sys.stderr)
        status["available"] = True
        return SkillExtractor(nlp, SKILL_DB, PhraseMatcher), status
    except (ImportError, ModuleNotFoundError, OSError, Exception) as exc:
        print(
            f"SkillNER unavailable ({type(exc).__name__}: {exc}). Falling back to taxonomy-only extraction.",
            file=sys.stderr,
        )
        status["available"] = False
        status["backend"] = "taxonomy_only"
        status["error"] = f"{type(exc).__name__}: {exc}"
        return None, status


# ---------------------------------------------------------------------------
# Candidate extraction + merging (Stages 2/3/5)
# ---------------------------------------------------------------------------


def find_evidence_sentence(term: str, sentence_objects: list[dict[str, str]]) -> tuple[str, str]:
    term_norm = canonicalize(term)
    for s in sentence_objects:
        if term_norm and term_norm in canonicalize(s["text"]):
            return s["text"], s["section"]
    if sentence_objects:
        return sentence_objects[0]["text"], sentence_objects[0]["section"]
    return "", ""


def extract_skillner_candidates(
    annotation: dict,
    taxonomy: Taxonomy,
    sentence_objects: list[dict[str, str]],
    id_to_type: dict[str, str],
    name_to_type: dict[str, str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    results = annotation.get("results", {})

    for bucket in ("full_matches", "ngram_scored"):
        for item in results.get(bucket, []):
            raw = (
                item.get("doc_node_value")
                or item.get("skill_name")
                or item.get("skill")
                or item.get("text")
            )
            if not raw:
                continue

            raw_norm = canonicalize(str(raw))
            if (
                len(raw_norm) <= 2
                and raw_norm not in SHORT_TOKEN_ALLOWLIST
            ):
                continue
            if (
                not raw_norm
                or not is_valid(raw_norm)
                or raw_norm in LOCAL_JUNK_STOPLIST
            ):
                continue

            canonical = taxonomy.normalize(raw_norm)
            canonical_flat = canonical.replace(" ", "")
            if (
                len(canonical_flat) == 1
                and canonical not in SHORT_TOKEN_ALLOWLIST
            ):
                continue
            if canonical in GENERIC_SINGLE_TOKEN_DROP:
                continue

            provisional_type = "unknown"
            skill_id = item.get("doc_node_id") or item.get("skill_id")
            if skill_id is not None:
                provisional_type = id_to_type.get(str(skill_id), "unknown")
            if provisional_type == "unknown":
                provisional_type = name_to_type.get(canonical, "unknown")
            # Relaxed recall guard:
            # allow multi-word non-taxonomy skills from SkillNER unless in junk lists.
            if canonical not in taxonomy.entries_by_skill and len(canonical.split()) == 1:
                if canonical not in {"python", "sql", "java", "stata", "spss", "r", "c++"}:
                    continue

            if canonical in seen:
                continue
            seen.add(canonical)

            evidence, section = find_evidence_sentence(raw_norm, sentence_objects)
            skill_type = provisional_type
            if skill_type == "unknown":
                skill_type = name_to_type.get(canonical, "unknown")
            if skill_type == "unknown":
                skill_type = taxonomy.get_meta(canonical).get("skill_type", "unknown")

            out.append(
                {
                    "canonical_skill": canonical,
                    "raw_span": str(raw).strip(),
                    "source_method": "SkillNER",
                    "evidence_sentence": evidence,
                    "evidence_section": section,
                    "similarity": None,
                    "skill_type": skill_type,
                }
            )

    return out


def extract_taxonomy_candidates(text: str, taxonomy: Taxonomy, sentence_objects: list[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    for canonical, pattern in taxonomy.alias_patterns:
        if not pattern.search(text):
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        evidence, section = find_evidence_sentence(canonical, sentence_objects)
        out.append(
            {
                "canonical_skill": canonical,
                "raw_span": canonical,
                "source_method": "taxonomy",
                "evidence_sentence": evidence,
                "evidence_section": section,
                "similarity": None,
                "skill_type": taxonomy.get_meta(canonical).get("skill_type", "unknown"),
            }
        )
    return out


def merge_candidates(
    skillner_cands: list[dict[str, Any]],
    taxonomy_cands: list[dict[str, Any]],
    taxonomy: Taxonomy,
    sections: dict[str, Any],
    config: PipelineConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    def ensure(skill: str) -> dict[str, Any]:
        if skill not in merged:
            meta = taxonomy.get_meta(skill)
            merged[skill] = {
                "canonical_skill": skill,
                "parent_category": meta.get("parent_category"),
                "skill_type": meta.get("skill_type", "unknown"),
                "methods": set(),
                "evidence": [],
                "raw_spans": set(),
                "max_similarity": 0.0,
            }
        return merged[skill]

    for cand in skillner_cands + taxonomy_cands:
        skill = cand["canonical_skill"]
        rec = ensure(skill)
        rec["methods"].add(cand["source_method"])
        if cand.get("raw_span"):
            rec["raw_spans"].add(cand["raw_span"])
        if cand.get("evidence_sentence"):
            rec["evidence"].append(
                {
                    "sentence": cand["evidence_sentence"],
                    "section": cand.get("evidence_section", "description"),
                }
            )
        if cand.get("similarity"):
            rec["max_similarity"] = max(rec["max_similarity"], float(cand["similarity"]))

        if rec["skill_type"] in {"unknown", ""} and cand.get("skill_type"):
            rec["skill_type"] = cand["skill_type"]

    title_norm = canonicalize(sections.get("title", ""))
    outcomes_norm = canonicalize(sections.get("learning_outcomes", ""))

    final: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for skill, rec in merged.items():
        methods = rec["methods"]
        max_sim = rec["max_similarity"]

        score = 0.0
        if "SkillNER" in methods:
            score += 0.62
        if "taxonomy" in methods:
            score += 0.34

        if skill in title_norm:
            score += 0.10
        if skill in outcomes_norm:
            score += 0.08

        evidence_sentence = rec["evidence"][0]["sentence"] if rec["evidence"] else ""
        sentence_norm = canonicalize(evidence_sentence)
        if any(v in sentence_norm for v in ACTION_VERBS):
            score += 0.08

        if skill in GENERIC_SKILL_PENALTY and "SkillNER" not in methods:
            score -= 0.18

        score = max(0.0, min(1.0, score))

        source_method = "both" if len(methods) >= 2 else next(iter(methods))
        rec_obj = {
            "raw_span": sorted(rec["raw_spans"])[0] if rec["raw_spans"] else skill,
            "canonical_skill": skill,
            "source_method": source_method,
            "confidence_score": round(score, 4),
            "parent_category": rec["parent_category"],
            "skill_type": rec["skill_type"] if rec["skill_type"] in {"hard", "soft"} else "unknown",
            "evidence_sentence": evidence_sentence,
        }

        if score >= config.min_confidence or "SkillNER" in methods:
            final.append(rec_obj)
        else:
            dropped.append(rec_obj)

    final.sort(key=lambda x: x["confidence_score"], reverse=True)

    debug = {
        "counts": {
            "skillner": len(skillner_cands),
            "taxonomy": len(taxonomy_cands),
            "merged": len(merged),
            "kept": len(final),
            "dropped": len(dropped),
        },
        "dropped_candidates": dropped[:30],
    }
    return final, debug


def apply_zero_skill_recovery(
    title: str,
    description: str,
    existing_skills: list[dict[str, Any]],
    taxonomy: Taxonomy,
) -> list[dict[str, Any]]:
    if existing_skills:
        return existing_skills

    text = canonicalize(f"{title} {description}")
    recovered: list[dict[str, Any]] = []
    for pattern, candidates in ZERO_SKILL_RECOVERY_RULES:
        if not re.search(pattern, text):
            continue
        for skill in candidates:
            canonical = taxonomy.normalize(skill)
            meta = taxonomy.get_meta(canonical)
            recovered.append(
                {
                    "raw_span": skill,
                    "canonical_skill": canonical,
                    "source_method": "rule_recovery",
                    "confidence_score": 0.46,
                    "parent_category": meta.get("parent_category"),
                    "skill_type": meta.get("skill_type", "unknown") if meta.get("skill_type") in {"hard", "soft"} else "unknown",
                    "evidence_sentence": "",
                }
            )
        break

    # Deduplicate and cap recovery to avoid inflation.
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in recovered:
        k = item["canonical_skill"]
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
        if len(out) >= 2:
            break

    # Secondary conservative recovery for rich but non-technical descriptions.
    # Applies only when description has enough signal and no explicit skills found.
    if out:
        return out

    words = text.split()
    if len(words) < 20:
        return out

    generic_candidates: list[str] = []
    if re.search(r"analy[sz]e|critical|evaluate|assess|argument|theory|criticism", text):
        generic_candidates.append("critical thinking")
    if re.search(r"research|inquiry|literature|scholarly|method", text):
        generic_candidates.append("research methods")
    if re.search(r"write|written|thesis|essay|report|discourse", text):
        generic_candidates.append("writing")
    if re.search(r"present|presentation|pitch|communicat", text):
        generic_candidates.append("communication")
    if re.search(r"design|develop|create|studio", text):
        generic_candidates.append("design thinking")

    # Keep at most 2 generic tags to avoid inflation.
    for skill in generic_candidates:
        canonical = taxonomy.normalize(skill)
        if canonical in seen:
            continue
        seen.add(canonical)
        meta = taxonomy.get_meta(canonical)
        out.append(
            {
                "raw_span": skill,
                "canonical_skill": canonical,
                "source_method": "rule_recovery",
                "confidence_score": 0.42,
                "parent_category": meta.get("parent_category"),
                "skill_type": meta.get("skill_type", "unknown") if meta.get("skill_type") in {"hard", "soft"} else "unknown",
                "evidence_sentence": "",
            }
        )
        if len(out) >= 2:
            break

    return out


# ---------------------------------------------------------------------------
# Main extraction flow
# ---------------------------------------------------------------------------


def extract_course_skills(
    input_csv: Path,
    output_rows_csv: Path,
    output_skills_csv: Path,
    taxonomy_path: str | None,
    config: PipelineConfig,
) -> dict[str, Any]:
    df = pd.read_csv(input_csv)
    required = {"module code", "title", "description"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")

    taxonomy = load_taxonomy(taxonomy_path)
    extractor, skillner_status = build_skillner_extractor()
    id_to_type, name_to_type = _load_skillner_type_lookup()

    semantic_status = {
        "enabled": False,
        "available": False,
        "model_name": None,
        "error": "disabled_by_design_skillner_plus_taxonomy",
    }

    diagnostics: dict[str, Any] = {
        "pipeline_checks": {
            "skillner": skillner_status,
            "taxonomy_matcher": {"enabled": True, "available": True},
            "semantic_matcher": semantic_status,
        },
        "run_stats": {
            "modules_total": int(len(df)),
            "unique_text_units": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "skillner_annotate_calls": 0,
            "skillner_failures": 0,
            "taxonomy_match_calls": 0,
            "semantic_calls": 0,
            "semantic_sentence_count": 0,
            "skillner_candidates_total": 0,
            "taxonomy_candidates_total": 0,
            "semantic_candidates_total": 0,
            "final_skills_total": 0,
            "modules_zero_skills": 0,
            "source_method_counts": {"SkillNER": 0, "taxonomy": 0, "embedding": 0, "both": 0},
        },
    }

    # Cache expensive extraction by shared description + outcomes text.
    extraction_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    rows_out: list[dict[str, Any]] = []
    pairs_out: list[dict[str, Any]] = []

    metadata_cols = [
        c for c in ("department", "faculty", "module_credit", "fulfill_requirements", "Undergraduate/Graduate")
        if c in df.columns
    ]

    for i, (_, row) in enumerate(df.iterrows(), 1):
        sections = extract_sections(row)
        text_key = "||".join([
            sections["description_for_extraction"],
            sections["learning_outcomes"],
        ])

        if text_key not in extraction_cache:
            diagnostics["run_stats"]["cache_misses"] += 1
            text_for_extract = " ".join(
                x for x in [sections["description_for_extraction"], sections["learning_outcomes"]] if x
            ).strip()

            # Hybrid design: always run taxonomy matcher.
            # SkillNER is attempted independently and contributes additional
            # high-precision candidates when available.
            ann: dict[str, Any] = {"results": {"full_matches": [], "ngram_scored": []}}
            if extractor is not None:
                try:
                    diagnostics["run_stats"]["skillner_annotate_calls"] += 1
                    ann = extractor.annotate(text_for_extract)
                except Exception:
                    diagnostics["run_stats"]["skillner_failures"] += 1

            skillner_cands = extract_skillner_candidates(
                annotation=ann,
                taxonomy=taxonomy,
                sentence_objects=sections["sentences"],
                id_to_type=id_to_type,
                name_to_type=name_to_type,
            )
            diagnostics["run_stats"]["skillner_candidates_total"] += len(skillner_cands)

            diagnostics["run_stats"]["taxonomy_match_calls"] += 1
            taxonomy_cands = extract_taxonomy_candidates(
                text=text_for_extract,
                taxonomy=taxonomy,
                sentence_objects=sections["sentences"],
            )
            diagnostics["run_stats"]["taxonomy_candidates_total"] += len(taxonomy_cands)

            semantic_cands: list[dict[str, Any]] = []

            base_candidates, debug_obj = merge_candidates(
                skillner_cands=skillner_cands,
                taxonomy_cands=taxonomy_cands,
                taxonomy=taxonomy,
                sections=sections,
                config=config,
            )
            extraction_cache[text_key] = (base_candidates, debug_obj)
        else:
            diagnostics["run_stats"]["cache_hits"] += 1

        base_candidates, cached_debug = extraction_cache[text_key]

        # Re-score title/outcome specific boosts per module to preserve section awareness.
        # We reuse merge output and only apply a lightweight title/outcome refinement.
        title_norm = canonicalize(sections["title"])
        outcomes_norm = canonicalize(sections["learning_outcomes"])

        final_candidates: list[dict[str, Any]] = []
        for c in base_candidates:
            score = float(c["confidence_score"])
            skill = c["canonical_skill"]
            if skill in title_norm:
                score = min(1.0, score + 0.05)
            if skill in outcomes_norm:
                score = min(1.0, score + 0.04)
            enriched = dict(c)
            enriched["confidence_score"] = round(score, 4)
            final_candidates.append(enriched)

        final_candidates = apply_zero_skill_recovery(
            title=sections["title"],
            description=sections["description_for_extraction"],
            existing_skills=final_candidates,
            taxonomy=taxonomy,
        )
        final_candidates.sort(key=lambda x: x["confidence_score"], reverse=True)
        diagnostics["run_stats"]["final_skills_total"] += len(final_candidates)
        if not final_candidates:
            diagnostics["run_stats"]["modules_zero_skills"] += 1
        for c in final_candidates:
            src = c.get("source_method", "unknown")
            if src not in diagnostics["run_stats"]["source_method_counts"]:
                diagnostics["run_stats"]["source_method_counts"][src] = 0
            diagnostics["run_stats"]["source_method_counts"][src] += 1

        skills = [c["canonical_skill"] for c in final_candidates]
        hard_skills = [c["canonical_skill"] for c in final_candidates if c["skill_type"] == "hard"]
        soft_skills = [c["canonical_skill"] for c in final_candidates if c["skill_type"] == "soft"]
        unknown_skills = [c["canonical_skill"] for c in final_candidates if c["skill_type"] == "unknown"]

        row_obj = {
            "module code": row["module code"],
            "title": sections["title"],
            "description": sections["description"],
            "description_for_extraction": sections["description_for_extraction"],
            "learning_outcomes": sections["learning_outcomes"],
            "skills": json.dumps(skills),
            "hard_skills": json.dumps(hard_skills),
            "soft_skills": json.dumps(soft_skills),
            "unknown_skills": json.dumps(unknown_skills),
            "skill_count": len(skills),
            "extracted_skills": json.dumps(final_candidates),
        }
        for col in metadata_cols:
            row_obj[col] = row[col]

        if config.debug:
            row_obj["debug_info"] = json.dumps(cached_debug)

        rows_out.append(row_obj)

        for cand in final_candidates:
            pair = {
                "module code": row["module code"],
                "title": sections["title"],
                "skill": cand["canonical_skill"],
                "skill_type": cand["skill_type"],
                "parent_category": cand.get("parent_category"),
                "source_method": cand["source_method"],
                "confidence_score": cand["confidence_score"],
                "evidence_sentence": cand.get("evidence_sentence", ""),
            }
            for col in metadata_cols:
                pair[col] = row[col]
            pairs_out.append(pair)

        if i % 500 == 0 or i == len(df):
            print(f"Processed {i}/{len(df)} modules...", file=sys.stderr)

    rows_df = pd.DataFrame(rows_out)
    rows_df.to_csv(output_rows_csv, index=False)

    pairs_df = pd.DataFrame(pairs_out)
    if not pairs_df.empty:
        pairs_df = pairs_df.drop_duplicates(subset=["module code", "skill"])
    pairs_df.to_csv(output_skills_csv, index=False)
    diagnostics["run_stats"]["unique_text_units"] = len(extraction_cache)
    return diagnostics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hybrid non-LLM skill extraction pipeline: SkillNER + taxonomy grounding + "
            "rule-based reranking (no embeddings)."
        )
    )
    parser.add_argument("--input", default="data/processed/cleaned_modules.csv")
    parser.add_argument("--output-rows", default="data/processed/modules_with_skills.csv")
    parser.add_argument("--output-skills", default="data/processed/module_skill_pairs.csv")
    parser.add_argument(
        "--taxonomy-path",
        default=None,
        help="Optional custom taxonomy file (.json or .csv).",
    )
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--diagnostics-out",
        default="data/processed/extraction_run_diagnostics.json",
        help="Path to write run diagnostics JSON (set empty string to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        min_confidence=args.min_confidence,
        debug=args.debug,
        diagnostics_out=args.diagnostics_out,
    )

    diagnostics = extract_course_skills(
        input_csv=Path(args.input),
        output_rows_csv=Path(args.output_rows),
        output_skills_csv=Path(args.output_skills),
        taxonomy_path=args.taxonomy_path,
        config=cfg,
    )

    if cfg.diagnostics_out:
        out_path = Path(cfg.diagnostics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
        print(f"Saved diagnostics: {cfg.diagnostics_out}")

    print("Method checks:")
    print(
        f"  SkillNER available={diagnostics['pipeline_checks']['skillner']['available']} "
        f"backend={diagnostics['pipeline_checks']['skillner']['backend']}"
    )
    print(
        f"  Semantic enabled={diagnostics['pipeline_checks']['semantic_matcher']['enabled']} "
        f"available={diagnostics['pipeline_checks']['semantic_matcher']['available']} "
        f"model={diagnostics['pipeline_checks']['semantic_matcher']['model_name']}"
    )
    print(f"  Semantic note: {diagnostics['pipeline_checks']['semantic_matcher']['error']}")
    print(
        "  Candidate totals "
        f"(SkillNER/Taxonomy/Semantic): "
        f"{diagnostics['run_stats']['skillner_candidates_total']}/"
        f"{diagnostics['run_stats']['taxonomy_candidates_total']}/"
        f"{diagnostics['run_stats']['semantic_candidates_total']}"
    )

    print(f"Saved: {args.output_rows}")
    print(f"Saved: {args.output_skills}")


if __name__ == "__main__":
    main()
