import pandas as pd
from src.config import JOBS_FILTERED

df = pd.read_csv(JOBS_FILTERED)

print(f"Total rows: {len(df)}")
print(f"Rows with #NAME?: {(df['clean_description'] == '#NAME?').sum()}")
print(f"Rows with garbled text: {df['clean_description'].str.contains('ðŸ', na=False).sum()}")
print(f"Rows with empty description: {df['clean_description'].isna().sum()}")