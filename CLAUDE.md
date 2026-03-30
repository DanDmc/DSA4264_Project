# CLAUDE.md — Project Context for Claude Code

## Who is the user?
Data scientist (NUS student) at MOE working on a university project (DSA4264).
Collaborating with a team. Claude Code assists with code, methodology design, and validation strategy.

---

## Project Goal
Assess how well NUS university courses prepare students for real-world jobs.

**Primary research question**: Do NUS courses cover the skills demanded by the Singapore job market?

**Three measurable sub-questions**:
1. What skills does the job market demand, and how frequently?
2. What skills do NUS courses teach?
3. How well do course-taught skills map onto job-demanded skills?

**Target audience**: MOE officers who want to make evidence-based decisions about curriculum relevance.

---

## Data Sources
- **NUSMods API** — course descriptions (AY2025–2026), fetched programmatically
- **MyCareersFuture job postings** — Jan 2025 to Jan 2026, Singapore labour market

---

## Two Core Methodologies

### Methodology 1 — Skill-Space Alignment (current focus)
Extract an explicit skill vocabulary from both course descriptions and job postings using SkillNer, then measure overlap.

Key metric: **DWSC (Demand-Weighted Skill Coverage)** — fraction of job-demanded skills covered by courses, weighted by posting frequency.

### Methodology 2 — Semantic Alignment (to be built)
Embed course descriptions and job descriptions using `sentence-transformers`, compute cosine similarity at the bullet/sentence level. Does not require an explicit skill vocabulary — captures coverage that SkillNer misses.

---

## Pipeline Structure (Course Skills)

```
NUSMods API
  │  step 1: process_nusmods.py
  ▼
data/processed/modules_raw.csv
data/processed/cleaned_modules.csv      ← undergraduate only (levels 1–4, excl 5000/6000)
  │  step 2: extract_course_skills.py
  ▼
data/processed/modules_with_skills.csv  ← one row per module, skills as list
data/processed/module_skill_pairs.csv   ← one row per module-skill pair
  │  step 3: postprocess_skills.py
  ▼
data/processed/modules_with_skills_clean.csv
  │  step 4 (optional): compare_extractors.py
  ▼
results/extractor_comparison.csv
results/extractor_comparison_summary.csv
```

**Single entry point** — run the whole pipeline or individual steps:
```bash
python -m src.course_skills_pipeline                  # steps 1 2 3
python -m src.course_skills_pipeline --steps 2 3      # skip fetch
python -m src.course_skills_pipeline --steps 4 \
    --ground-truth data/validation/annotations.xlsx   # run validation
```

All paths are configured in `Config` at the top of `src/course_skills_pipeline.py`.

---

## Module Filtering Rules (process_nusmods.py)
- **Keep**: modules with a valid, non-placeholder description AND level 1000–4000
- **Remove**: 5000-level and 6000-level (graduate courses) — detected by first digit of numeric part of module code
- **Remove**: modules with placeholder/empty descriptions
- No filtering by module credit (MC) — this was replaced by level filtering
- No `Undergraduate/Graduate` column — graduate courses are filtered out entirely
- No `fulfill_requirements` column

Columns in `cleaned_modules.csv`: `module code`, `title`, `description`, `description_clean`, `department`, `faculty`, `module_credit`

---

## Skill Definition (for annotation and extraction)
> **A skill is something the text explicitly states the course teaches, that is specific enough to represent a distinct concept a student could have or not have.**

Key rules for annotation:
- **Explicit only** — do not infer skills from context. If the text doesn't say it, don't label it.
- **Academic terms count** — e.g. "lower and upper bounds", "prune-and-search" are valid skills even if they don't appear in job ads. The coverage analysis step handles job-market relevance separately.
- **Topics count as skills** — SkillNer doesn't distinguish, job ads list topic knowledge as requirements. Use `knowledge_type` to capture the distinction.
- **"Understanding of X" → X is theoretical** — the framing verb makes the whole sentence theoretical. Still label the named skills.
- **Action verbs → applied** — e.g. "implement", "derive", "verify", "select", "design". Conceptual verbs → theoretical — e.g. "covers", "introduces", "provides understanding of".
- **Prerequisites are not skills** — if a sentence says "students are expected to know X", label it `sentence_type = prerequisite`, not taught.

---

## Manual Annotation Schema
Excel file with 4 sheets:

| Sheet | Purpose |
|---|---|
| `courses_annotated` | One row per skill per course — the ground truth |
| `course_metadata` | One row per course — code, title, discipline cluster, module level |
| `skillner_comparison` | SkillNer raw output on the same 40 courses (populated after annotation) |
| `annotation_guide` | Reference card with rules and examples |

Columns in `courses_annotated`:

| Column | Values |
|---|---|
| `module_code` | e.g. CS3230 |
| `skill_label` | Normalized skill name (lowercase, singular) |
| `skill_type` | `hard` / `soft` |
| `sentence_type` | `taught` / `tool` / `prerequisite` |
| `knowledge_type` | `theoretical` / `applied` |
| `source_sentence` | The exact sentence from the description the skill came from |
| `notes` | Optional — edge case reasoning |

---

## Validation Strategy (Methodology 1)

### Why validate on course descriptions only?
SkillNer was trained on job postings. Its recall on job descriptions is expected to be high by design. The unknown is how well it generalises to academic course language. **Validate on course descriptions; only spot-check job descriptions.**

### Sample: 40 courses, stratified
- **8 disciplinary clusters**: Computing, Data Science & Statistics, Business & Finance, Engineering, Mathematics, Social Sciences, Life Sciences, Humanities & Arts
- **4 module levels**: 1000, 2000, 3000, 4000 (10 per level, spread across clusters)
- Selection prioritises coverage across degree programmes (CS, DS, ECE, Econ, Biz, etc.)

### Ground truth annotation
- ~160 skill entries across 40 courses (~4 per course average)
- Annotated manually by the team following the skill definition and annotation schema above
- Each annotator should annotate independently, then reconcile disagreements

### Extractor methods to compare (step 4)

| Method ID | Name | Description |
|---|---|---|
| 1 | `skillner_only` | SkillNer output with no post-filtering |
| 2 | `skillner_taxonomy` | SkillNer + taxonomy/canonicalization (current production pipeline) |
| 3 | `llm_fewshot` | LLM with few-shot examples from manual annotations |
| 4 | `hybrid` | SkillNer generates candidates, LLM filters/accepts |

### Evaluation metrics (per method, per course)
- **Precision** — of extracted skills, what fraction match ground truth
- **Recall** — of ground truth skills, what fraction were extracted
- **F1** — harmonic mean (primary metric — both FP and FN are costly)
- Breakdown by: `skill_type`, `knowledge_type`, `sentence_type`, module level, discipline cluster

### Fuzzy matching for evaluation
Exact string match after canonicalization (lowercase, singularize). If a skill span is within a ground truth entry (or vice versa), count as a match. Edge cases handled by embedding similarity (cosine > 0.85 on `sentence-transformers`).

### LLM access for methods 3/4
No paid API currently. Recommended free option: **Groq** (free tier, fast, supports Llama 3.1 70B).
```bash
pip install groq
export GROQ_API_KEY=your_key_here  # from console.groq.com
```
The pipeline auto-detects the provider from environment variables.

---

## Planned Improvements (not yet built)
- **IDF weighting for DWSC** — so common/generic skills don't dominate the coverage metric
- **Theory-practice ratio** — per-course operationalisability score (fraction of skills that are `applied`)
- **Semantic alignment pipeline** — `sentence-transformers` bullet-level matching (Methodology 2)
- **Sanity test suite** — end-to-end tests for the extraction pipeline

---

## Important Constraints
- Large processed CSVs (e.g. job skills files > 100MB) are gitignored via `data/processed/*.csv` — do not commit them.
- `skill_classification/hard_soft_skills.py` is outside `src/` — this is a known inconsistency, do not move it without updating all imports.
- `canonicalize()` and `_SYNONYM_MAP` are duplicated between `postprocess_skills.py` and `hard_soft_skills.py` — they must be kept in sync. Do not add synonyms to one without adding to the other.
