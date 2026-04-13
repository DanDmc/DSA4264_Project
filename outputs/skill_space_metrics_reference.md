# Skill-Space Alignment — Metrics Reference
## Methodology 1: Explicit Skill Vocabulary Matching

---

## Core Metrics

| # | Component | What it is | How it is computed | Breakdowns | What it tells MOE |
|---|---|---|---|---|---|
| 1 | **Baseline SCR** (reference) | Exact string match coverage — fraction of job-demanded skills that appear verbatim in NUS course skill lists | `n_exact_covered / n_unique_job_skills` | Per job category | Lower bound. No synonym or paraphrase matching. Included for comparison only — not a finding in itself. |
| 2 | **Soft-IDF-SCR** (main metric) | Fraction of job-demanded skills covered by NUS courses, weighted by how distinctive those skills are across job categories | Exact match = 1.0 credit; NN cosine similarity ≥ θ=0.72 = partial credit. Scores IDF-weighted across 43 job categories. Global aggregate = weighted mean by posting volume. | Per job category; per SSOC minor group | "What share of what the job market actually cares about does NUS cover?" Skills rare across categories receive higher IDF weight — prevents generic skills from dominating. |
| 2a | ↳ Applied split | Soft-IDF-SCR restricted to job skills matched by an *applied* course skill | Same as #2; NN credit only if the matched course skill is labelled applied | Per job category | "Does NUS cover job-demanded skills in a hands-on, practical way?" |
| 2b | ↳ Theoretical split | Soft-IDF-SCR restricted to job skills matched by a *theoretical* course skill | Same as #2; NN credit only if the matched course skill is labelled theoretical | Per job category | Shows how much of NUS coverage is conceptual rather than practical. |
| 3 | **Soft-SCR** (SSOC-aligned) | Fraction of target-occupation job skills covered by each programme's course skill set | Denominator filtered to SSOC minor groups mapped to each major / faculty / department. Numerator = course skills with exact or NN match (≥ θ) in that filtered set. No IDF weighting — enables fair cross-programme comparison. | Per major (+ core/elective split); per faculty; per department | "How well does this specific programme cover the skills expected in its intended career track?" |
| 4 | **Priority Gap Index (PGI)** | Which job categories have the largest, most urgent uncovered skill gaps | `PGI = job_demand_weight × gap_rate` where `gap_rate = 1 − soft_idf_scr` and `job_demand_weight = category_job_mentions / total_job_mentions`. Confidence flagged by bootstrap CI width. | Per job category (ranked); action band assigned | "Where should MOE focus curriculum investment?" High PGI = high-demand sector + low NUS coverage. |
| 5 | **Foundational Ratio** | Share of NUS theoretical skills that have no equivalent in job postings, broken down by the reason | `n_foundational / n_theoretical_course_skills`. Each foundational skill classified by its nearest-neighbour similarity to any job skill: **Vocabulary Gap** (NN sim 0.50–0.72 — concept exists in job market under a different name) vs **Truly Foundational** (NN sim < 0.50 — no close job equivalent). | Per faculty; global band split | "How much of NUS teaching is genuinely academic content vs. a vocabulary mismatch with industry?" A Vocabulary Gap is a communication problem; a Truly Foundational skill is intentional academic content. |
| 6 | **Degree Alignment Score** | Which degree programmes are most misaligned with their target career tracks, combining skill-space and semantic evidence | `alignment_score = 0.5 × sem_pctile + 0.5 × skill_pctile` where both are percentile-ranked (0–100) across all 55 degrees. Skill metric = SSOC-aligned Soft-SCR (Metric 3). Semantic metric = Targeted SSOC Top-K Mean Similarity (Methodology 2). Both restricted to SSOC-mapped jobs only. | Per degree (ranked 1–55); scatter plot | "Which NUS degrees should MOE review first?" Low score = poor alignment on both skill-space and semantic dimensions against the same target occupations. |

---

## MOE Output Angles

| # | Output | What it is | What it tells MOE |
|---|---|---|---|
| A1 | **Action Matrix** | Each job category assigned to an action band: *Act Now / Monitor / Encourage / Maintain* | Derived from PGI band and confidence flag. Gives a direct triage for curriculum investment decisions. |
| A2 | **Coverage Lift** | How much Soft-IDF-SCR improved over Baseline SCR per category | Large lift = vocabulary mismatch between courses and job postings, not a genuine curriculum gap. Small lift = skills are genuinely missing. |
| A4 | **Top Skill Targets** | Top uncovered skills per job category, ranked by job demand frequency | The specific skills NUS is missing. Directly actionable input for syllabus review by category. |

---

## Results Outputs

All outputs are written to `DATA_ROOT/results/` (OneDrive, not committed to git).

| Output folder | Key file | Key columns |
|---|---|---|
| `breakdown_analysis_{source}/` | `job_category.csv` | `category`, `baseline_scr`, `soft_idf_scr`, `soft_idf_scr_applied`, `soft_idf_scr_theoretical`, `n_unique_job_skills`, `n_genuine_gaps` |
| | `major.csv` | `major`, `relevant_ssoc_minors`, `n_relevant_job_skills`, `soft_scr_all`, `soft_scr_applied`, `soft_scr_theoretical`, `soft_scr_all_core` |
| | `faculty.csv` | `faculty_final`, `relevant_ssoc_minors`, `n_relevant_job_skills`, `soft_scr_all`, `soft_scr_applied`, `soft_scr_theoretical` |
| | `department.csv` | `dept_final`, `relevant_ssoc_minors`, `n_relevant_job_skills`, `soft_scr_all`, `soft_scr_applied`, `soft_scr_theoretical` |
| | `ssoc_minor.csv` | `ssoc_minor_title`, `soft_idf_scr`, `baseline_scr`, `n_unique_job_skills`, `n_genuine_gaps` |
| `policy_priority_{source}_t{t}/` | `priority_gap_index_category_level.csv` | `category`, `priority_gap_index`, `gap_rate`, `job_demand_weight`, `soft_idf_scr`, `confidence_flag`, `ci_width` |
| | `priority_gap_index_top10_per_category.csv` | Top 10 uncovered skills per category with demand frequency |
| | `confidence_flags_by_category.csv` | `category`, `confidence_flag`, `ci_width` |
| `foundational_layer_{source}/` | `foundational_theoretical_skills_no_job_demand.csv` | `skill_canon`, `relevance_band`, `nearest_job_skill`, `nearest_similarity_pct`, `interpretation`, `n_modules`, `n_faculties`, `faculties` |
| | `foundational_ratio_by_faculty.csv` | `faculty`, `foundational_ratio`, `n_total_unique_skills`, `n_theoretical_unique_skills`, `n_foundational_theoretical_only_skills` |
| | `summary_foundational_layer.json` | Global counts + relevance band split (vocabulary_gap n/%, truly_foundational n/%) |
| `moe_angles_{source}_t{t}/` | `angle1_category_action_matrix.csv` | `category`, `soft_idf_scr`, `priority_gap_index`, `priority_band`, `confidence_flag`, `recommended_action`, `action_rationale` |
| | `angle2_coverage_lift_depth_confidence.csv` | `category`, `baseline_scr_exact`, `improved_soft_idf_scr`, `absolute_lift_pp`, `relative_lift_pct` |
| | `angle4_top_skill_targets_per_category.csv` | `category`, `skill`, `job_freq`, `gap_score` |
| `combined_degree_priority/` | `combined_degree_priority.csv` | `priority_rank`, `major`, `degree_faculty`, `sem_pctile`, `skill_pctile`, `alignment_score`, `targeted_top_k_mean`, `soft_scr_all`, `n_mapped_jobs`, `relevant_ssoc_minors` |
| | `fig_degree_priority_scatter.png` | Scatter plot of all 55 degrees — semantic alignment percentile (x) vs skill coverage percentile (y) |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| θ = 0.72 (NN match threshold) | Calibrated via threshold sweep (range 0.60–0.85). Validated against robustness pipeline — category rankings stable across threshold variants. |
| IDF over 43 job categories | Prevents generic skills (e.g. "communication", "teamwork") from dominating the global score. Skills rare across categories are more diagnostic of specific curriculum gaps. |
| SSOC-aligned denominator for Soft-SCR | A CS programme should not be penalised for not covering nursing skills. Denominator filtered to the occupational groups each programme trains for via a manually curated major→SSOC mapping. MOE should validate this mapping before operationalising the metric. |
| No IDF weighting on course-side (Soft-SCR) | IDF weights are derived from the distribution across 43 job categories. Applying them within a single programme's skill set is not meaningful. |
| Core/elective split at major level only | Module type is reliably populated only at major level via the degree-module mapping. Faculty and department breakdowns use all modules. |
| Vocabulary Gap threshold = 0.50 | Below 0.50, the nearest job skill is semantically distant enough to represent a different concept. 0.50–0.72 captures conceptual overlap where the same idea is described with different vocabulary in academic vs. industry contexts. |
| 50-50 weight for Degree Alignment Score | Equal weight between skill-space and semantic evidence. Both metrics are restricted to the same SSOC-mapped jobs, making the comparison fair. Rankings are stable under 60-40 weighting in either direction. |
| Degree Alignment Score uses percentile ranks | Raw metrics are on different scales (cosine similarity ~0.49–0.68 vs Soft-SCR 0–0.13). Percentile ranking normalises both to 0–100 before combining, preventing one metric from dominating. |

---

## Known Limitations

| Limitation | Affected outputs | Recommended treatment |
|---|---|---|
| Life Sciences: only 3/77 modules have skills extracted | `major.csv` (Soft-SCR = 0), `combined_degree_priority.csv` (rank inflated) | Exclude from course-side tables in report; flag as data limitation |
| Engineering Science: low skill extraction coverage | `major.csv`, `combined_degree_priority.csv` | Flag alongside Life Sciences |
| Major→SSOC mapping is manually curated, not independently validated | All SSOC-aligned outputs (Metrics 3, 6) | Present with explicit caveat; recommend MOE validation |
| `skills_list` job source uses NLP-extracted skills — some noise | All outputs | Robustness pipeline confirms headline findings hold; noise affects absolute numbers more than rankings |
| 50-50 weight in Degree Alignment Score is a design choice | `combined_degree_priority.csv` | Sensitivity check: rerun with 60-40 weights and verify top-10 degrees are unchanged |
