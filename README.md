# DSA4264 Project - Are NUS courses preparing graduates for "good" jobs?

An NLP project analysing the alignment between NUS university courses and Singapore job postings, framed for Ministry of Education (MOE) policy officers. We use semantic embedding similarity and LLM-based skill extraction to quantify how well NUS degree programmes prepare graduates for entry-level "good" jobs.

## Project Structure

```
repo root/
├── src/
│   ├── config.py                        ← centralised paths, parameters & constants
│   ├── data_processing/
│   │   ├── process_jobs.py              ← step 1a: parse raw job JSONs → CSV
│   │   ├── clean_jobs.py                ← step 1b: HTML→text, encoding fixes
│   │   ├── ssoc_mapping.py              ← step 2: SSOC 2024 hierarchy enrichment
│   │   ├── final_jobs_filtering.py      ← step 3: filter to entry-level roles
│   │   └── process_nusmods.py           ← fetch & clean NUS modules from API
│   ├── embedding/
│   │   ├── embed.py                     ← BGE embedding pipeline (whole-text mode)
│   │   └── colab_runner.py              ← Colab notebook helper for GPU embedding
│   ├── analysis/
│   │   ├── similarity_analysis.py       ← full similarity metric pipeline (9 stages)
│   │   ├── report_outputs.py            ← generate figures for the technical report
│   │   ├── skills_analysis.py           ← skill-based alignment analysis
│   │   └── query_tool.py               ← CLI tool to query module/degree/SSOC/job alignment
│   ├── pipelines/
│   │   ├── 01_jobs_processing.py        ← end-to-end jobs pipeline (steps 1a→3)
│   │   └── course_skills_pipeline.py    ← end-to-end course skill extraction & validation
│   ├── skills/                          ← skill extraction scripts (SkillNER, LLM few-shot)
│   └── validation/
│       ├── compare_extractors.py        ← skill extractor method comparison & F1 scoring
│       ├── validation_dataset.xlsx      ← manually curated embedding validation pairs
│       └── labelling_sheet.xlsx         ← human labelling sheet for embedding model validation
├── notebooks/                           ← preliminary EDA and filtering justifications
├── outputs/
│   ├── jobs_filtering_visualisations/   ← charts justifying experience & salary thresholds
│   ├── report_figures/                  ← publication-ready figures for the technical report
│   └── similarity_analysis_outputs/     ← all analysis CSVs, metadata JSON
├── validation/                          ← skill extraction validation results & Excel outputs
├── archive/                             ← deprecated / superseded scripts (reference only)
├── reports/                             ← technical report (Markdown, rendered via MkDocs)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Data Storage

Large datasets live in a **shared OneDrive folder** (`DSA4264_Project_Data/`), not in git. Each team member syncs the folder locally and sets a `DATA_ROOT` path in their `.env`.

```
DSA4264_Project_Data/
├── raw/
│   ├── jobs/
│   │   └── 20260125_20260131/               ← batch of raw job posting JSONs
│   ├── courses/                             ← NUSMods API output (modules_raw.csv)
│   ├── Degree_Mapping_Final.xlsx            ← manually curated degree → module mapping
│   ├── major_ssoc_mapping.csv               ← manually curated degree → SSOC minor mapping
│   └── ssoc2024-detailed-definitions.xlsx   ← official SSOC 2024 hierarchy definitions
├── processed/
│   ├── jobs/
│   │   ├── 01a_jobs_extracted.csv
│   │   ├── 01b_jobs_cleaned.csv
│   │   ├── 02_jobs_ssoc_mapped.csv
│   │   └── 03_jobs_filtered.csv             ← final: 7,101 entry-level "good" jobs
│   └── courses/
│       ├── modules_cleaned.csv              ← 8,615 undergrad modules with valid descriptions
│       ├── module_skill_pairs.csv
│       └── degree_module_mapping.csv        ← exploded degree → module lookup
├── embeddings/                              ← BGE bge-large-en-v1.5 embeddings (.npy + index CSVs)
└── results/
    ├── similarity_matrix.npy                ← full module × job cosine similarity matrix
    └── skills/                              ← skill extraction outputs
```

## Getting Started

### 1. Clone the repo

```bash
git clone <repo-url>
cd DSA4264
```

### 2. Set up Python environment

```bash
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
# .venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Configure data path

The shared OneDrive folder is here: [DSA4264_Project_Data](https://1drv.ms/f/c/e79fc0eb716d4f63/IgA2417HTDQZSbcrL56eLuzMAe-b-Y1NGFcNVDyP294fyz4?e=gkmbuy). Add it to your OneDrive so it syncs locally, then:

```bash
cp .env.example .env
# Edit .env and set DATA_ROOT to your local OneDrive path, e.g.:
# DATA_ROOT=C:\Users\YourName\OneDrive\...\DSA4264_Project_Data
```

All data paths and configurable parameters are centralised in `src/config.py` — no paths are hardcoded elsewhere.

## Running the Pipelines

### Jobs processing

```bash
python -m src.pipelines.01_jobs_processing           # full pipeline (steps 1a → 1b → 2 → 3)
python -m src.pipelines.01_jobs_processing --steps 2 3  # run specific steps only
```

Steps: parse raw JSONs → text cleaning → SSOC mapping → filter to entry-level roles (≤2 years experience, ≥$3,500/month). Outputs land in `processed/jobs/` on OneDrive. New job batches can be added by placing JSONs in a new `raw/jobs/YYYYMMDD_YYYYMMDD/` subfolder — the pipeline auto-scans all batch subfolders and tracks provenance via a `source_batch` column.

### Courses processing

```bash
python -m src.data_processing.process_nusmods
```

Fetches all modules from the NUSMods API, filters to undergraduate modules with valid descriptions (excludes levels 5–6, placeholder descriptions, internship modules). Outputs `modules_raw.csv` and `modules_cleaned.csv`.

### Embedding

Embeddings are computed on Google Colab (GPU) using `src/embedding/embed.py` with the BGE `bge-large-en-v1.5` model. Asymmetric instruction prefixes are applied for course-to-job retrieval. Outputs (`.npy` arrays + index CSVs) are saved to `embeddings/` on OneDrive.


### Query tool

```bash
python -m src.analysis.query_tool module CS3244
python -m src.analysis.query_tool degree "Computer Science"
python -m src.analysis.query_tool ssoc "SOFTWARE AND APPLICATIONS DEVELOPERS"
python -m src.analysis.query_tool job "data analyst"
python -m src.analysis.query_tool job 12345 --k 20
```

Interactive CLI for spot-checking alignment results by module, degree, SSOC group, or job posting.

## Key Parameters

All configurable in `src/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `MAX_YEARS_EXPERIENCE` | 2 | Max years experience for entry-level job filter |
| `MIN_SALARY_AVG` | 3500 | Min average salary threshold ($SGD/month) |
| `EXCLUDED_MODULE_LEVELS` | {5, 6} | Graduate module levels to exclude |
| `EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | Sentence embedding model (1024-dim) |
| `ANALYSIS_TOP_K` | 10 | Top-K job matches per module/degree |
| `ANALYSIS_BREADTH_SSOC_LEVEL` | `ssoc_minor_title` | SSOC hierarchy level for breadth counting |
| `ANALYSIS_DEGREE_AGG_TOP_N` | 10 | Top-N modules averaged per job for degree aggregation |
| `ANALYSIS_COVERAGE_THRESHOLD` | `None` (auto) | Coverage threshold; auto = mean + 1 SD |
