#just a final filter for jobs min years expereince <=2 to keep data manageable and focus on fresh grad roles for this project

import pandas as pd

# Load the dataset
input_path = "data/processed/jobs_processed_ssoc_mapped.csv"
output_path = "data/processed/final_jobs_processed_filtered.csv"

df = pd.read_csv(input_path)

# Filter rows where minimum_years_experience <= 2
filtered_df = df[df["minimum_years_experience"] <= 2]

# Save the filtered dataset
filtered_df.to_csv(output_path, index=False)

print(f"Filtered data saved to {output_path}")