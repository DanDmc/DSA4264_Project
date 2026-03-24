import json

import pandas as pd

INPUT_PATH = "data/processed/filtered_jobs_with_skills.csv"
OUTPUT_PATH = "data/processed/filtered_job_skill_pairs.csv"

df = pd.read_csv(INPUT_PATH)
df["skills"] = df["skills"].apply(json.loads)

job_skill_pairs = (
    df[["job_id", "title", "skills"]]
    .explode("skills")
    .rename(columns={"skills": "skill"})
)

job_skill_pairs.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(job_skill_pairs)} rows to {OUTPUT_PATH}")
