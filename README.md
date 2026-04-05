# DSA4264 — Does University Education Prepare Students for the Workforce?

An NLP project analysing the alignment between NUS modules and Singapore job postings, framed for Ministry of Education (MOE) policy officers.

## Project Structure

```
repo root/
├── src/
│   ├── config.py                      ← shared paths & parameters (reads from .env)
│   ├── pipelines/
│   │   └── 01_jobs_processing.py      ← runs the full jobs pipeline (steps 1–3)
│   ├── data_processing/
│   │   ├── process_jobs.py            ← step 1: parse raw job JSONs
│   │   ├── ssoc_mapping.py            ← step 2: map SSOC codes to level hierarchy and descriptions
│   │   ├── final_jobs_filtering.py    ← step 3: filter to entry-level roles (default ≤2 years experience)
│   │   └── process_nusmods.py         ← fetch & clean NUS modules from API
│   ├── skills/                        ← skill extraction scripts
│   ├── embedding/                     ← embedding pipeline
│   ├── analysis/                      ← metric analysis (similarity scores & skills coverage)
│   └── validation/                    ← e.g. skill extractor comparison scripts
├── notebooks/                         ← exploratory analysis
├── validation/                        ← validation data & results
├── outputs/                           ← analysis results, charts, summary CSVs
├── reports/                           ← final report (Markdown) & key figures
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Data Storage

Large datasets are stored in a **shared OneDrive folder**, not in this repo. Each team member syncs the folder locally and configures their path via `.env`.

```
OneDrive: DSA4264_Project_Data/
├── raw/
│   ├── jobs/
│   │   └── 20260125_20260131/         ← provided batch of job posting JSONs
│   ├── courses/                       ← NUSMods API output
│   └── ssoc2024-detailed-definitions.xlsx
├── processed/
│   ├── jobs/
│   │   ├── 01_jobs_parsed.csv
│   │   ├── 02_jobs_ssoc_mapped.csv
│   │   └── 03_jobs_filtered.csv
│   └── courses/
│       └── modules_cleaned.csv
└── embeddings/
```

## Getting Started

### 1. Clone the repo

```
git clone <repo-url>
cd DSA4264
```

### 2. Set up Python environment

```
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
.venv\Scripts\activate             # Windows
pip install -r requirements.txt
```

### 3. Configure data path

The shared OneDrive folder is here: [DSA4264_Project_Data](https://1drv.ms/f/c/e79fc0eb716d4f63/IgA2417HTDQZSbcrL56eLuzMAe-b-Y1NGFcNVDyP294fyz4?e=gkmbuy). Add it to your OneDrive so it syncs locally, then use that local path in your `.env`.

Copy `.env.example` in the repo root, rename the copy to `.env`, and fill in your local OneDrive path to the shared folder, e.g. `DATA_ROOT=C:\Users\YourName\OneDrive\...\DSA4264_Project_Data`

## Running the Pipelines

### Jobs processing

```
python -m src.pipelines.01_jobs_processing
```

Runs 3 steps sequentially: parse raw JSONs → SSOC mapping → filter to entry-level roles. Use `--steps 2 3` to run specific steps only. Outputs land in `processed/jobs/` in the OneDrive folder.

### Courses processing

```
python -m src.data_processing.process_nusmods
```

Fetches modules from the NUSMods API, filters to undergraduate modules with valid descriptions. Outputs `modules_raw.csv` to `raw/courses/` and `modules_cleaned.csv` to `processed/courses/`.

## Configuration

All data paths and pipeline parameters are centralised in `src/config.py`. Key configurable parameters:

| Parameter | Default | Description |
|---|---|---|
| `MAX_YEARS_EXPERIENCE` | 2 | Maximum years of experience for job filtering |
| `EXCLUDED_MODULE_LEVELS` | {5, 6} | Module levels to exclude (graduate courses) |
| `EXCLUDED_DESCRIPTION_KEYWORDS` | ["internship"] | Keywords that trigger module exclusion |
| `NUSMODS_API_URL` | AY2025-2026 | NUSMods API endpoint |

## Adding New Job Posting Batches

To add a new month of job postings:

1. Create a new batch subfolder in the OneDrive data folder: `raw/jobs/YYYYMMDD_YYYYMMDD/`
2. Place the JSON files in it
3. Rerun the jobs pipeline — it automatically scans all batch subfolders

Each job record includes a `source_batch` column so you can trace which batch it came from.
