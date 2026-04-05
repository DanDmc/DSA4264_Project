import pandas as pd

INPUT_PATH = "data/processed/jobs_processed_with_skills_full.csv"
OUTPUT_PATH = "data/processed/filtered_jobs_with_skills.csv"

OUTPUT_COLUMNS = [
    "job_id",
    "title",
    "clean_description",
    "job_text",
    "skillner_clean_description",
    "skills",
    "hard_skills",
    "soft_skills",
    "certifications",
    "skill_count",
]

df = pd.read_csv(INPUT_PATH)
df_filtered = df[
    (df["minimum_years_experience"] <= 2)
    & (df["skill_count"] != 0)
]
df_relevant_columns = df_filtered[OUTPUT_COLUMNS]
df_relevant_columns.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df_relevant_columns)} rows to {OUTPUT_PATH}")
