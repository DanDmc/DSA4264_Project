# Similarity Analysis — Metrics Reference

## Core Metrics

| # | Component | What it is | How it is computed | Breakdowns | What it tells MOE |
|---|-----------|------------|--------------------|------------|-------------------|
| **1** | **Similarity Matrix** (base layer) | Pairwise semantic similarity between every module and every job | Cosine similarity of BGE-large embeddings (L2-normalised, so dot product = cosine). Produces an (n_modules × n_jobs) matrix. | — | Foundation for all downstream metrics. Not reported directly. |
| **2** | **Top-K Mean Similarity** | Strength of a module's or degree's best job-market alignment | For each unit of analysis, select the K jobs with highest similarity, take the mean of those K scores. K=10 by default (configurable). | See 2a–2d below | "How strongly does this module/degree connect to its best-matching jobs?" |
| 2a | ↳ Module-level | Per-module alignment score | Top-K mean over the module's row in the sim matrix | Per module (ranked) | Identifies which individual courses have strongest/weakest job relevance |
| 2b | ↳ Degree-level | Per-degree alignment score | For each job, compute mean similarity of top-N (N=10) most similar modules in the degree → gives a per-job score → then top-K mean across jobs | Per degree (ranked) | Identifies which degree programmes best prepare students for the job market |
| 2c | ↳ SSOC-group alignment | Which occupational groups are best/worst served by the curriculum | For each SSOC minor group, compute mean similarity of each module to jobs in that group → report top-N modules per group | Per SSOC minor group | Reveals which occupational sectors have strong/weak curriculum coverage |
| 2d | ↳ Faculty / Department / Degree-faculty summaries | Aggregated alignment scores at institutional grouping levels | Mean and median of top-K scores across modules or degrees within each group | Per faculty, per department, per degree-faculty | High-level institutional view — which faculties are most/least aligned |
| **3** | **Alignment Breadth** | Diversity of job categories a module/degree is relevant to | Count the number of distinct SSOC minor groups represented in the unit's top-K job matches | See 3a–3b below | "Is this module/degree a specialist (few SSOC groups) or generalist (many)?" Complements top-K similarity — high score + narrow breadth = strong niche; moderate score + wide breadth = versatile |
| 3a | ↳ Module-level breadth | Per-module breadth | Count distinct SSOC minor groups among module's top-K matched jobs | Per module (ranked) | Identifies specialist vs. generalist courses |
| 3b | ↳ Degree-level breadth | Per-degree breadth | Count distinct SSOC minor groups among degree's top-K matched jobs | Per degree (ranked) | Shows which degrees open doors to diverse career paths |
| **4** | **Job Market Coverage** | Proportion of all entry-level jobs a module/degree is meaningfully aligned with | Count jobs where similarity ≥ threshold, divide by total jobs. Threshold = global mean + 1 SD of the full similarity distribution (dataset-relative, not absolute). | See 4a–4b below | "What share of the entry-level job market can graduates of this degree realistically target?" |
| 4a | ↳ Module-level coverage | Per-module coverage % | % of all jobs above threshold for that module's row | Per module (ranked) | Shows individual course market reach |
| 4b | ↳ Degree-level coverage | Per-degree coverage % | For each job, take mean of top-N modules → % of jobs where this score exceeds threshold | Per degree (ranked) | Key employability signal — "doing this degree gives you a reasonable shot at X% of entry-level roles" |
| **5** | **Targeted SSOC Alignment** | How well a degree aligns with its expected occupational groups vs. the general job market | Using curated major→SSOC minor mapping: for each degree, compute top-K mean similarity against only jobs within its mapped SSOC groups. Compare against the degree's overall top-K score. | Per degree (with mapped SSOC groups) | "Is this degree well-focused on its intended career tracks, or are its strongest matches in unexpected fields?" Higher targeted-to-overall ratio suggests good curricular focus. |

## Supporting Analyses

| # | Component | What it is | How it is computed | Breakdowns | What it tells MOE |
|---|-----------|------------|--------------------|------------|-------------------|
| **S1** | **Similarity Distribution Stats** | Statistical profile of the full sim matrix | Mean, SD, min, max, key percentiles (P90, P95, P99) of all similarity scores | Global | Informs threshold selection for coverage. Reported in methodology, not as a finding. |
| **S2** | **Threshold Stability Check** | Robustness test for coverage rankings | Compute degree-level coverage at mean+0.5SD, mean+1SD, mean+1.5SD; compare rankings via Spearman correlation | Global (3 threshold variants) | Demonstrates that coverage rankings are not sensitive to exact threshold choice. Reported in methodology. |

## Design Decisions (for report)

| Decision | Rationale |
|----------|-----------|
| K = 10 (configurable) | Captures strongest alignment signals while being robust to noise from individual postings. Rankings stable across K = 5, 10, 20. |
| Degree aggregation = mean of top-10 modules per job | Approximates a realistic student course load's relevance to a specific job. More realistic than max (too generous, one outlier module dominates) or full mean (too harsh, diluted by irrelevant modules). |
| Threshold = mean + 1 SD | Dataset-relative; identifies pairs with "meaningfully above average" similarity (~top 16% of all pairs). No claim of absolute meaning. |
| Breadth counted at SSOC minor level | Represents meaningfully distinct career domains. More granular levels would overcount within-family variation. |
| Coverage percentages are relative, not absolute | Valid for ranking degrees against each other within this dataset. Not comparable across different models, time periods, or universities. |
| Major–SSOC mapping is manually curated | Not independently validated. Presented with caveat that MOE should validate the mapping before operationalising this metric. |
