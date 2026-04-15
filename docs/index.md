---
title: DSA4264 Technical Report
tags: [onboarding]

---

---
title: DSA4264 Technical Report
tags: [onboarding]

---
# DSA4264 Technical Report
![image](https://hackmd.io/_uploads/H1ANDIpn-l.png)


## 1. Context

Rapid advancements in technology, particularly in artificial intelligence and digitalisation, have reshaped labour market demands. Employers increasingly prioritise job-relevant skills over traditional academic qualifications, placing pressure on higher education institutions to remain aligned with industry needs. In Singapore, this is evident with around 22.5% of workers reporting a mismatch between their skills and job requirements, while approximately 24% of employers report skills gaps in their workforce (Heng, 2026).

Within the Ministry of Education (MOE), university curricula are reviewed through consultations with industry stakeholders. However, these processes are subject to time lags due to the rapidly evolving economy. Reliance on aggregate indicators such as employment rates also provides limited granularity, making it difficult to identify specific gaps between course content and job requirements. As a result, curriculum updates may struggle to keep pace with fast-changing industries.

This project explores how data-driven methods can complement existing approaches by providing a more scalable and systematic way to assess course–job alignment, enabling more timely and granular insights into the relationship between education and labour demand.

## 2. Scope

### 2.1 Problem

MOE currently lacks a scalable and systematic way to evaluate how well university courses prepare students for the workforce. Existing approaches, such as the Graduate Employment Survey (GES) and periodic industry conferences, have several limitations.

First, they are largely reactive, reflecting outcomes months or years after graduation and offering limited forward-looking insights. Second, they operate at a high level of aggregation, lacking the granularity needed to capture variation across courses, disciplines, and job roles. As a result, they do not provide a systematic understanding of how course content aligns with labour market demands, limiting policymakers’ ability to identify specific areas of misalignment or implement targeted curriculum improvements.

This creates a significant information gap for MOE, limiting its ability to assess curriculum relevance and respond to evolving skill demands. Given that Singapore produces over 16,000 graduates annually, even small mismatches can scale into substantial workforce inefficiencies.

The impact extends across stakeholders. Graduates face longer job searches, while employers incur additional training costs of approximately SGD 500 to SGD 1,000 per employee annually (HRSINGAPORE, 2024). At the national level, misalignment reduces the efficiency of public education spending, which is approximately 2.2% of GDP (NCEE). 
Data science provides a suitable approach to address these gaps by enabling scalable analysis of job postings and course descriptions. This allows for systematic identification of misalignment and emerging demand patterns that are difficult to capture through traditional methods.

### 2.2 Success Criteria

From a business perspective, the goal is to provide a robust and interpretable measure of alignment between university courses and labour market demands, at both the skill and content level. A successful outcome would enable MOE to systematically identify areas of misalignment across programmes, including underrepresented skills and gaps in course content.

These insights support evidence-based, forward-looking policy decisions in curriculum design and workforce planning, and guide the prioritisation of curriculum updates and resource allocation.

From an operational perspective, the system should produce reliable, consistent, and interpretable alignment metrics, including achieving ≥85% precision in skill extraction and alignment when evaluated against manually annotated samples.

In addition, the system should ensure standardised representation of skills and course content, reducing ambiguity and improving consistency and comparability of results across different courses and job categories.
Success is further demonstrated if the alignment results align with external benchmarks, such as expert evaluation or relevant employment indicators, indicating that the system produces credible and policy-relevant insights.

### 2.3 Assumptions

#### Data Assumptions
Job advertisements are assumed to be a reasonable proxy for labour market demand, although they may contain generic or inflated requirements. Similarly, course descriptions are assumed to reflect the skills and knowledge acquired by students, despite being high-level and not fully capturing depth or variation.
The dataset is also assumed to be representative of broader labour market trends, despite being limited in time scope. If the data does not reflect longer-term patterns, the resulting insights may be biased or incomplete.

#### Technical assumptions
The approach assumes that semantic similarity in embedding space provides a meaningful proxy for alignment between job requirements and course content. In particular, it assumes that embeddings from both domains are directly comparable within the same vector space, despite differences in style, abstraction, and intent.
It also assumes that aggregating similarity scores across skills or text components yields a valid overall measure of alignment, even though this may simplify complex relationships such as differences in depth, importance, or contextual relevance.

#### Application assumptions
The project assumes that policymakers and educational institutions have the capacity to act on the insights generated. In practice, institutional and resource constraints may limit the speed and extent of implementation.

## 3. Methodology

### 3.1 Technical Assumptions 

#### Definitions
This project examines the relationship between NUS undergraduate courses and “good” entry-level jobs using two complementary analyses: semantic similarity and skills overlap. Skills are further categorised into applied and theoretical skills.

| Term               | Definition |
|--------------------|------------|
| Entry-level job    | Job postings requiring ≤2 years of experience exhibit a low proportion of senior responsibilities (~10–20%), approximating roles accessible to fresh NUS graduates![image](https://hackmd.io/_uploads/Hk4OU86hbg.png)
|
| Good job           | Job postings with ≥ S$3,500 average monthly salary, aligned with the lowest 25th percentile amongst NUS courses in Graduate Employment Survey outcomes (MOE, 2025). Chosen as a reasonable proxy for desirable graduate employment. |
| Skill              | Discrete capability extracted from course or job text for consistent cross-domain matching. |
| Applied skills     | Action-oriented skills (e.g., implement, design, analyse) representing job-executable competencies. |
| Theoretical skills | Conceptual skills (e.g., understanding, introduces, covers) representing foundational knowledge. |

#### Feature Availability 
Course data was sourced from the NUSMods API (AY2025–2026), while job postings were obtained from MyCareersFuture (25–31 January 2026). While largely reliable, employer-provided skills_list may be inconsistent or incomplete due to self-reporting. Unfortunately, useful information such as course syllabus were not available.
| Dataset | Raw Features |
|--------|----------|
| **Courses (NUSMods)** | course code, title, description, description_clean, department, faculty, course_credit |
| **Jobs (MyCareersFuture)** | job_id, title, raw_description, skills_list, minimum_years_experience, ssoc_code, category, salary_min, salary_max, salary_avg, posting_date |

#### Hypothesis
As an exploratory NLP study, no formal statistical hypotheses are tested. The analysis is guided by two assumptions:
(1) semantic similarity and skills overlap capture complementary dimensions of alignment (topical relevance vs competency coverage), and
(2) alignment differs meaningfully across degrees, faculties, and SSOC groups, motivating disaggregated analysis for policy relevance.

#### Computational Resources
All experiments were run on local CPUs or Google Colab (T4 GPU). LLM-based skill extraction used free-tier APIs (Cerebras, Groq, Gemini, Ollama).

### 3.2 Data

#### Collection
Data was sourced from the NUSMods API and MyCareersFuture (25–31 January 2026). SSOC mapping was performed using the official Singapore Standard Occupational Classification dataset (Department of Statistics Singapore, n.d.). A degree-to-course mapping dataset was manually constructed using official NUS websites. Given the complexity and frequent updates to course requirements, this should be interpreted as an approximate guide for mapping rather than a definitive source of truth.

#### Cleaning
Course data required minimal preprocessing due to structured API formatting. Job postings required extensive cleaning, including HTML removal, standardisation to plain text, and filtering of non-informative content such as boilerplate equal-opportunity statements, URLs, emails, emojis, and non-Latin text. Job titles and descriptions were then concatenated into a unified text field.

#### Filtering
Only undergraduate courses at levels 1000–4000 were retained (NUS Registrar, 2026), with courses lacking meaningful descriptions or classified as internships removed, resulting in 8,615 courses.

![Course level](outputs/filter_justification/Course level.jpeg)

Job postings were filtered to ≤2 years of experience and ≥S$3,500 average monthly salary, consistent with the project’s definitions of entry-level and “good” jobs. After filtering, 7,101 postings (31.3% of the dataset) were retained.

![image](https://hackmd.io/_uploads/S1bnULp2bl.png)


![image](https://hackmd.io/_uploads/rJ8pULph-g.png)


### 3.3 Experimental Design 

### 3.3.1 Semantic Similarity

### Embedding and Validation
We used a semantic embedding approach to measure alignment between NUS courses and job postings using BGE-large (BAAI/bge-large-en-v1.5). This model was selected for its strong performance on semantic retrieval benchmarks (e.g. MTEB) and its ability to[](https://) incorporate instruction prefixes, which is useful for aligning academic content with job descriptions (Abimbola et al., 2026; Xiao et al., 2023). 

To determine the most effective text representation strategy, we evaluated four embedding variants: (V1) full-text embedding, (V2) sentence-level embedding with noise filtering based on cosine similarity to the document centroid, (V3) sentence mean pooling, and (V4) sentence-weighted pooling using centroid similarity-based weights. These variants were designed to test whether sentence-level decomposition improves robustness over full-document embeddings.

A validation set was manually constructed from 30 jobs and 18 courses across six occupational categories (e.g. software engineering, accounting, legal), yielding 540 labelled pairs. Category membership was subjectively assigned based on human judgement of course–job relevance within each domain, and used as proxy ground truth.
Performance was evaluated using three metrics:

Mean Reciprocal Rank (MRR): measures how highly the first same-category course is ranked for each job (higher = better ranking quality).

Separation gap: difference in mean cosine similarity between same-category and cross-category pairs, measuring class discriminability.

Misalignment rate: proportion of jobs where the top-ranked course belongs to a different category, capturing top-1 classification error.

Results showed that the full-text embedding approach (V1) consistently outperformed sentence-based methods, achieving the highest MRR and strongest class separation. As a result, V1 was selected for downstream analysis.

| Variant | Method | MRR ↑ | Separation Gap ↑ | Misalignment ↓ 
|--------|--------|------|------------------|---------------|
| V1 (Baseline) | Full-text embedding | **0.9289** | **0.1056** | **0.0222** |
| V2 | Sentence + threshold | 0.8456 | 0.0735 | 0.0622 |
| V3 | Sentence mean-pool | 0.8253 | 0.0688 | 0.0844 |
| V4 | Sentence weighted | 0.8281 | 0.0700 | 0.0844 |

To further validate embedding quality, a blinded human evaluation was conducted on 48 course–job pairs spanning high-, mid-, and low-similarity cases. Human relevance scores (0–2 scale) were moderately to strongly correlated with cosine similarity scores (Spearman ρ = 0.69, p < 0.001), indicating that embedding similarity reflects meaningful semantic alignment.

Embeddings were computed using Google Colab (T4 GPU). All job postings with ≤2 years of experience were embedded prior to salary filtering to preserve a richer semantic space; the “good jobs” constraint (≥S$3,500/month) was applied later by masking the similarity matrix during analysis.

### Metrics

Based on the computed similarity scores, the following metrics were explored for analysis.

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Top-10 mean similarity** | Average cosine similarity between each course and its 10 most similar jobs. | Captures strongest course–job matches. Higher values indicate stronger alignment.|
| **Breadth** | Number of distinct occupational groups in the top-10 job matches. | Measures diversity of job alignment. Higher values indicate broader applicability. |
| **Coverage** | Percentage of jobs above a similarity threshold (mean + 1 standard deviation). | Measures overall market reach. Higher values indicate wider alignment. |

### 3.3.2 Skill-Space Alignment

#### Skill Extraction and Validation

Two extraction methods were evaluated: SkillNer, a rule-based NER tool trained on job postings, and LLM few-shot extraction. Fine-tuned NER was not considered due to limited labelled data (39 courses, ~160 instances), which would likely lead to overfitting. Embeddings were not evaluated as an extraction method, as they operate on pre-identified spans rather than extracting skills from raw text.

To validate both approaches, 39 course descriptions were sampled across 8 disciplinary clusters and 4 course levels to ensure coverage of varying content types. Skills were manually annotated for each description to construct a ground truth dataset. Extracted skills were matched against ground truth using using a two-stage matching approach, consisting of exact matching followed by partial (substring-based) matching to account for minor variations in phrasing. Precision, recall, and F1-score were then computed. 

The LLM extractor achieved F1 = 0.565, outperforming SkillNer which underperformed on academic-style terminology.

![image](https://hackmd.io/_uploads/ryykPUp3Wl.png)


The LLM approach was therefore selected as the production extractor, with its F1 score representing a known ceiling on extraction reliability and should be considered when interpreting coverage results.

#### Algorithms
Metric development followed an iterative process: a baseline was established, its limitations diagnosed via vocabulary mismatch analysis, and improvements introduced to address these gaps.

A baseline SCR using exact string matching was established as a reproducible lower bound (macro SCR = 37.6%). Analysis revealed that 55.2% of initially uncovered job skills are near-misses — semantically equivalent skills described differently across academic and industry text — indicating that exact matching understates true coverage.

This motivated the transition to semantic matching. The primary metric, Soft-IDF-SCR, incorporates embedding-based similarity (Reimers & Gurevych, 2019) and IDF weighting (Sparck Jones, 1972), raising macro coverage to 69.3% — a 31.7 percentage point improvement. [include fig_baseline_vs_improved.png]

A null model permutation test (B=1,000 iterations) further validated that NUS's coverage pattern is statistically non-random, with 41 out of 43 job categories exceeding the null distribution (z = 14.92, p ≈ 0. [include fig_actual_vs_null.png]


#### Evaluation
Soft-IDF-SCR is selected as the primary metric. Plain SCR was rejected as it treats a skill in 5 postings identically to one in 5,000, misrepresenting true market priorities. DW-SCR partially addresses this but overweights generic skills (e.g., “communication”, “teamwork”), masking domain-specific gaps. Pure cosine similarity was rejected as it produces a non-interpretable scalar score without identifying specific skill gaps. 

Soft-IDF-SCR addresses all three: demand-weighting ensures proportional contribution; IDF amplifies domain-specific signals; semantic matching recovers vocabulary mismatches, producing interpretable outputs aligned with MOE’s objective of identifying high-demand skills that NUS curricula fail to cover.

#### Hyperparameters
The cosine similarity threshold θ was calibrated empirically by sweeping values from 0.60 to 0.95 against 357 human-labelled course-job skill pairs, with θ = 0.67 maximising F1 = 0.715. [include fig_calibration_curve.png] A slightly conservative value of θ = 0.72 was adopted to reduce false positives at the cost of marginally lower recall, prioritising precision given policy implications. IDF log-smoothing (+1) was applied as regularisation, preventing ubiquitous skills from being zeroed out. Category imbalance is addressed by IDF weighting, normalising the signal across job categories. Bootstrap 95% confidence intervals (B=1,000) were computed per category to quantify uncertainty in coverage estimates, with the robustness sweep confirming category rankings remain stable across threshold variants. [include ​fig_threshold_sweep.png]


### Metrics

### Metrics

| Metric | Definition | Business Interpretation |
|---|---|---|
| **Baseline SCR** | Fraction of job-demanded skills with an exact string match to at least one course skill. | Strict lower bound on alignment; understates true coverage due to vocabulary mismatch. |
| **Soft-SCR** | Fraction of job-demanded skills with a semantic match (cosine similarity ≥ θ) to at least one course skill. | Better reflects alignment by recovering vocabulary mismatches. |
| **Soft-IDF-SCR** | Soft-SCR where each job skill is weighted by IDF — specialised skills count more than generic ones. | Highlights domain-specific unmet demand; penalises coverage of only generic skills. |
| **Foundational Ratio** | Fraction of a degree's course skills labelled theoretical rather than applied. | Indicates how practice-ready a programme's curriculum is. |
| **PGI (Priority Gap Index)** | A degree's relative job demand (job count / max job count across degrees) multiplied by its misalignment rate (1 − alignment score), where alignment score is the equal-weighted average of its semantic similarity and Soft-SCR percentile ranks. | Prioritises degrees for curriculum review by jointly capturing employer demand and misalignment. |


## 4. Findings

### 4.1 Results

This section presents key findings most relevant to MOE, focusing on experimental results, their business implications, and potential for production deployment.

### 4.1.1 Semantic Similarity

### Course and Degree-level Analysis

Top-10 Mean Similarity is computed differently at course and degree levels. At the course level, each module is directly compared to all jobs using cosine similarity, and the final score is the mean of the top-10 most similar jobs. At the degree level, each degree is represented as a set of modules, and for each job the similarity is computed using the top-10 most relevant modules, before averaging across the top-10 jobs. This provides a consistent but more granular evaluation framework across levels of aggregation.

Across both levels, results are as expected: professional, analytical, and quantitatively oriented programmes dominate the top ranks, while humanities and social science subjects appear in the lower ranks under the entry-level job filter. Notably, breadth tends to exhibit an inverse relationship with similarity, 


#### Course-Job Anlaysis
| Similarity Rank | Course | Top-10 Mean Similarity | Breadth | Coverage (%) |
|-----------------|--------|-----------|----------|--------------|
| 1 | Investment Analysis and Portfolio Management | 0.759 | 2 | 56.02 |
| 2 | Building Information Modeling for Project Management | 0.758 | 4 | 19.56 |
| 3 | Advanced Portfolio Management: Securities Analysis & Valuation | 0.743 | 3 | 83.95 |
| 4 | Analytical Tools for Consulting | 0.738 | 2 | 75.65 |
| 5 | Processing of Microelectronic Materials | 0.734 | 3 | 36.15 |
| ... | | | | |
| 6096 | Socrates on Trial | 0.433 | 7 | 0 |
| 6097 | Writing the Desert | 0.430 | 8 | 0 |
| 6098 | Ancient Western Political Thought | 0.418 | 4 | 0 |
| 6099 | Hinduism, Nationalism, and the Bhagavad Gita in the 20th Century | 0.417 | 5 | 0 |
| 6100 | American Political Thought | 0.407 | 6 | 0 |
#### Degree-Job Anlaysis
| Similarity Rank | Degree | Faculty | Top-10 Mean Similarity | Breadth | Coverage |
|-----------------|--------|---------|--------------------------|----------|----------|
| 1 | Civil Engineering | Design & Engineering | 0.682 | 2 | 37.1% |
| 2 | Finance | Business | 0.681 | 3 | 79.4% |
| 3 | Law | Law | 0.681 | 1 | 91.2% |
| 4 | Mechanical Engineering | Design & Engineering | 0.676 | 2 | 40.7% |
| 5 | Materials Science & Engineering | Design & Engineering | 0.676 | 3 | 36.1% |
| ... | | | | | |
| 51 | Political Science | Arts & Social Sciences | 0.576 | 5 | 13.7% |
| 52 | Anthropology | Arts & Social Sciences | 0.565 | 9 | 16.3% |
| 53 | Philosophy, Politics & Economics | Arts & Social Sciences | 0.553 | 6 | 7.2% |
| 54 | Philosophy | Arts & Social Sciences | 0.550 | 7 | 11.4% |
| 55 | English Literature | Arts & Social Sciences | 0.538 | 4 | 1.5% |

### Breadth-Similarity Trade-Off

### Coverage

### Query tool



### 4.1.2 Skills

#### Degree-To-Career Track Skill Coverage

Skill coverage by degree (SSOC-aligned Soft-SCR) is highly uneven, ranging from **19.3% (Law)** to **0.0% (Life Sciences\*)**. The top 3 programmes are **Law (19.3%)**, **Psychology (9.0%)**, and **Accountancy (8.6%)**; the bottom 3 are **Life Sciences\* (0.0%)**, **Engineering Science\* (0.3%)**, and **English Language and Linguistics (0.3%)**. Most degrees cluster at low single-digit coverage. Life Sciences and Engineering Science are flagged for low extraction coverage.

#### Foundational Ratio

[insert figure]

Foundational Ratio is highest in **Life Sciences (97%)**, **Philosophy (96%)**, **Sociology (87%)**, and **English Language and Linguistics (86%)**, and lowest in **Nursing (39%)**, **Industrial Design (44%)**, **Applied Business Analytics (44%)**, and **Operations and Supply Chain Management (45%)**.

#### Data Science and Analytics Deep-Dive

| Metric | Value |
|---|---:|
| DSA course skills (unique) | 342 |
| Demanded job skills (unique) | 1,192 |
| Exact match | 9 (0.8%) |
| Semantic match (`θ = 0.72`) | 6 (0.5%) |
| Genuine gap | 1,177 (98.7%) |
| Soft-SCR | 1.2% |

**Top 5 demanded skills:**

| Rank | Job skill | Postings | Status |
|---:|---|---:|---|
| 1 | python | 91 | EXACT |
| 2 | information technology | 87 | NON-EXACT |
| 3 | troubleshooting | 84 | NON-EXACT |
| 4 | sql | 74 | NON-EXACT |
| 5 | software development | 74 | NON-EXACT |

**Top 5 DSA skills with semantic matches:**

| Rank | DSA skill | Postings | Matched job skill |
|---:|---|---:|---|
| 1 | assessing the precision of estimates | 3 | estimates |
| 2 | scalar valued functions | 2 | scala |
| 3 | finance | 1 | asset finance |
| 4 | big data analysis | 1 | large scale data analysis |
| 5 | basic learning systems | 1 | learning management systems |

#### PGI Index

The **Priority Gap Index (PGI)** scores each degree as:

$$\text{PGI} = \text{Job Demand Weight} \times \text{Misalignment Rate}$$

where Job Demand Weight is the degree's mapped job count normalised by the highest-demand degree (0–1), and Misalignment Rate = 1 − Alignment Score / 100. The Alignment Score is a 50-50 blend of semantic similarity and skill-space coverage (Soft-SCR) percentile ranks.

**Engineering Science** ranks highest, followed by **Industrial & Systems Engineering**, **Electrical Engineering**, and **English Language and Linguistics**. The remaining top 15 are predominantly STEM and engineering disciplines. **Data Science and Analytics** and **Statistics** rank lower, reflecting comparatively better alignment.



### 4.2 Discussion




### 4.3 Recommendations













