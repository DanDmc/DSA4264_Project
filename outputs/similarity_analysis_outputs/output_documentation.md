# Similarity Analysis — Output File Documentation

All CSV outputs are saved to `REPO_ROOT/outputs/similarity_analysis_outputs/`.
The similarity matrix (.npy) is saved to OneDrive at `DATA_ROOT/results/similarity_analysis_results/`.


## Output Files Overview

| File | Rows | Purpose | Metrics Reference |
|------|------|---------|-------------------|
| `module_summary.csv` | 1 per module | Central module-level results: alignment strength, breadth, coverage | Metrics 2a, 3a, 4a |
| `degree_summary.csv` | 1 per degree | Central degree-level results: alignment strength, breadth, coverage | Metrics 2b, 3b, 4b |
| `top_k_matches_per_module.csv` | K per module | Detailed job matches for each module (long-form) | Metric 2a (detail) |
| `top_k_matches_per_degree.csv` | K per degree | Detailed job matches for each degree (long-form) | Metric 2b (detail) |
| `ssoc_minor_alignment.csv` | 1 per SSOC minor group | Which modules best serve each occupational group | Metric 2c |
| `targeted_ssoc_alignment.csv` | 1 per degree (with mapping) | How well degrees align with their intended SSOC groups | Metric 5 |
| `faculty_summary.csv` | 1 per faculty | Aggregated module stats by faculty | Metric 2d |
| `department_summary.csv` | 1 per department | Aggregated module stats by department | Metric 2d |
| `degree_faculty_summary.csv` | 1 per degree-faculty | Aggregated degree stats by faculty | Metric 2d |
| `analysis_metadata.json` | — | Run parameters, distribution stats, threshold stability | Supporting S1, S2 |
| `similarity_matrix.npy` (OneDrive) | — | Raw (modules × jobs) cosine similarity matrix (float16) | Metric 1 |


## Column Definitions by File

### module_summary.csv

One row per NUS module. The primary file for comparing individual courses.

| Column | Description |
|--------|-------------|
| `module_code` | NUS module code (e.g. CS3244) |
| `module_title` | Full module title |
| `module_faculty` | Module's home faculty |
| `module_department` | Module's home department |
| `top_k_mean_sim` | Mean cosine similarity to the module's K most similar jobs. Higher = stronger alignment to best-matching jobs. |
| `rank_top_k_sim` | Dense rank by top_k_mean_sim (1 = highest alignment) |
| `top_k_max_sim` | Similarity to the single most similar job (the best possible match) |
| `top_k_min_sim` | Similarity to the K-th most similar job (the weakest match within top-K) |
| `breadth_n_ssoc_groups` | Number of distinct SSOC minor groups represented in the module's top-K job matches. High = generalist, low = specialist. |
| `rank_breadth` | Dense rank by breadth (1 = broadest) |
| `coverage_n_jobs` | Count of jobs where the module's similarity exceeds the coverage threshold |
| `coverage_pct` | coverage_n_jobs as a percentage of all jobs |
| `rank_coverage` | Dense rank by coverage_pct (1 = widest market reach) |


### degree_summary.csv

One row per NUS degree (major). The primary file for comparing degree programmes.

| Column | Description |
|--------|-------------|
| `major` | Degree name (e.g. Computer Science) |
| `degree_faculty` | Faculty the degree belongs to |
| `n_modules_total` | Total modules listed in degree mapping (core + elective) |
| `n_modules_matched` | How many of those modules were found in the embedding index |
| `top_k_mean_sim` | Mean of the degree's K highest per-job scores. Each per-job score is computed as the mean of the top-N (default 10) most similar modules for that job. |
| `rank_top_k_sim` | Dense rank by top_k_mean_sim (1 = highest alignment) |
| `top_k_max_sim` | The single highest per-job score (best-matching job for this degree) |
| `overall_mean_sim` | Mean per-job score across all jobs (general market alignment) |
| `breadth_n_ssoc_groups` | Distinct SSOC minor groups in the degree's top-K job matches |
| `rank_breadth` | Dense rank by breadth (1 = broadest career diversity) |
| `coverage_n_jobs` | Jobs where the degree's per-job score exceeds the coverage threshold |
| `coverage_pct` | coverage_n_jobs as a percentage of all jobs |
| `rank_coverage` | Dense rank by coverage_pct (1 = widest market reach) |


### top_k_matches_per_module.csv

Long-form: one row per (module, job match rank). Shows exactly which jobs each module is most aligned with.

| Column | Description |
|--------|-------------|
| `module_code` | NUS module code |
| `module_title` | Module title |
| `module_faculty` | Module's home faculty |
| `module_department` | Module's home department |
| `rank` | Match rank within this module (1 = most similar job) |
| `job_id` | Job posting identifier |
| `job_title` | Job title text |
| `ssoc_code` | SSOC occupation code of the job |
| `ssoc_major_title` | SSOC major group (broadest level, ~10 groups). Included as metadata for context. |
| `ssoc_minor_title` | SSOC minor group (~30 groups). The level used for breadth counting and analytical breakdowns. |
| `ssoc_unit_title` | SSOC unit group (more granular). Included as metadata for drill-down inspection. |
| `similarity` | Cosine similarity score between module and job embeddings |


### top_k_matches_per_degree.csv

Long-form: one row per (degree, job match rank). Shows which jobs each degree is most aligned with.

| Column | Description |
|--------|-------------|
| `major` | Degree name |
| `degree_faculty` | Faculty the degree belongs to |
| `rank` | Match rank within this degree (1 = most similar job) |
| `job_id` | Job posting identifier |
| `job_title` | Job title text |
| `ssoc_minor_title` | SSOC minor group of the job |
| `ssoc_unit_title` | SSOC unit group of the job |
| `similarity` | Per-job alignment score (mean of top-N module similarities for this job) |
| `best_module` | The single module with highest individual similarity to this job (shows which course drove the match) |


### ssoc_minor_alignment.csv

One row per SSOC minor group. Shows which occupational categories are best/worst served by the NUS curriculum.

| Column | Description |
|--------|-------------|
| `ssoc_group` | SSOC minor group title (e.g. "SOFTWARE AND APPLICATIONS DEVELOPERS AND ANALYSTS") |
| `ssoc_major_title` | Parent SSOC major group for context |
| `n_jobs` | Number of jobs in this SSOC group |
| `overall_mean_sim` | Mean similarity of all modules to all jobs in this group. Higher = curriculum is generally more relevant to this occupation. |
| `rank_overall_mean_sim` | Dense rank (1 = SSOC group best served by curriculum) |
| `top1_module` / `top1_title` / `top1_sim` | Most aligned module and its similarity |
| `top2_module` / `top2_title` / `top2_sim` | Second most aligned module |
| `top3_module` / `top3_title` / `top3_sim` | Third most aligned module |


### targeted_ssoc_alignment.csv

One row per degree that has an entry in major_ssoc_mapping.csv. Compares each degree's alignment to its "expected" occupational groups vs. the overall job market.

| Column | Description |
|--------|-------------|
| `major` | Degree name |
| `degree_faculty` | Faculty |
| `mapped_ssoc_groups` | The SSOC minor groups this degree is expected to prepare students for (from curated mapping) |
| `n_mapped_jobs` | Number of jobs in the mapped SSOC groups |
| `targeted_top_k_mean` | Top-K mean similarity computed against only the mapped SSOC jobs |
| `rank_targeted` | Dense rank by targeted_top_k_mean (1 = strongest targeted alignment) |
| `overall_top_k_mean` | Top-K mean similarity against all jobs (same as in degree_summary.csv) |
| `targeted_vs_overall_ratio` | targeted_top_k_mean / overall_top_k_mean. Values >1 mean the degree is more aligned with its intended fields than the general market; <1 means its strongest matches are in unexpected fields. |
| `rank_ratio` | Dense rank by ratio (1 = most focused on intended fields) |


### faculty_summary.csv

One row per module home faculty. Aggregated from module_summary.csv.

| Column | Description |
|--------|-------------|
| `module_faculty` | Faculty name |
| `n_modules` | Number of modules in this faculty |
| `mean_top_k_sim` | Mean of top_k_mean_sim across modules in this faculty |
| `median_top_k_sim` | Median of top_k_mean_sim |
| `mean_breadth` | Mean breadth across modules |
| `mean_coverage_pct` | Mean coverage % across modules |


### department_summary.csv

Same structure as faculty_summary.csv, grouped by `module_department`.


### degree_faculty_summary.csv

Same structure as faculty_summary.csv, but aggregated from degree_summary.csv by `degree_faculty`. Shows how each faculty's degree programmes compare.

| Column | Description |
|--------|-------------|
| `degree_faculty` | Faculty name |
| `n_degrees` | Number of degrees in this faculty |
| `mean_top_k_sim` | Mean of top_k_mean_sim across degrees |
| `median_top_k_sim` | Median of top_k_mean_sim |
| `mean_breadth` | Mean breadth across degrees |
| `mean_coverage_pct` | Mean coverage % across degrees |


### analysis_metadata.json

Run metadata and distribution statistics. Not a CSV — a JSON file for programmatic access.

| Key | Description |
|-----|-------------|
| `run_timestamp` | When the analysis was run |
| `parameters.top_k` | K value used |
| `parameters.degree_agg_top_n` | N value used for degree aggregation |
| `parameters.coverage_threshold` | The similarity threshold used for coverage |
| `parameters.breadth_ssoc_level` | SSOC level used for breadth counting |
| `parameters.embedding_model` | Embedding model used |
| `data_shape` | Number of modules, jobs, and degrees |
| `distribution_stats` | Full similarity matrix statistics (mean, SD, percentiles) |
| `threshold_stability` | Spearman correlations of coverage rankings across 3 threshold variants (mean+0.5SD, mean+1SD, mean+1.5SD) |


## SSOC Levels Used

The SSOC hierarchy has 5 levels. This pipeline uses:

- **ssoc_minor_title** — the primary analytical level for breadth counting, SSOC group alignment, and targeted alignment. Represents meaningfully distinct career domains (~30 groups in our dataset). Examples: "SOFTWARE AND APPLICATIONS DEVELOPERS AND ANALYSTS", "FINANCE PROFESSIONALS".

- **ssoc_major_title** — included as a metadata column for context grouping. Too coarse for breakdowns (~10 groups).

- **ssoc_unit_title** — included as a metadata column in top-K match CSVs for drill-down inspection. Too granular for breadth counting.

- **ssoc_submajor_title** and **ssoc_occupation_title** — not used. Submajor adds little over minor; occupation is the most granular level and would make breadth counting meaningless with K=10.
