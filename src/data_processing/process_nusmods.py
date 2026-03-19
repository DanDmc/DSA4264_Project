import requests
import pandas as pd
import re

url = "https://api.nusmods.com/v2/2025-2026/moduleInfo.json"
output_path = "modules.csv"
cleaned_output_path = "cleaned_modules.csv"

def fetch_module_data(api_url):
    """Fetch raw module data from api"""
    response = requests.get(api_url, timeout = 10)
    response.raise_for_status()
    return response.json()

def extract_fields(data):
    """Extract selected fields from raw module data

    Expected structure of each item in fetched data:

    {
    "moduleCode": "ABM5001",
    "title": "Leadership in Biomedicine",
    "description": "Leadership is fundamental to the success of individuals and organizations. As you progress in your biomedicine career, you will have to lead individuals, teams and organizations. This course prepares you to lead, by equipping you with principles, skills and practices of leadership.",
    "moduleCredit": "2",
    "department": "NUS Medicine Dean's Office",
    "faculty": "Yong Loo Lin Sch of Medicine",
    "workload": [
        3,
        0,
        0,
        4,
        3
    ],
    "gradingBasisDescription": "Graded",
    "semesterData": [
        {
        "semester": 2,
        "covidZones": [
            "Unknown"
        ]
        }
    ]
    }
    """
    records = []

    for item in data:
        record = {
            "module code": item.get("moduleCode"),
            "title": item.get("title"),
            "description": item.get("description"),
            "department": item.get("department"),
            "faculty": item.get("faculty")
        }
        records.append(record)

    return records

def save_to_csv(records, output_path):
    """Save extracted records to a CSV file"""
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)


def _normalize_description(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return (
        str(value)
        .strip()
        .lower()
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def _is_valid_description(description: str) -> bool:
    normalized = _normalize_description(description)
    if not normalized:
        return False

    if "internship" in normalized:
        return False

    if normalized in {
        "not available",
        "not available.",
        "not applicable",
        "unrestricted elective",
        "nil",
        "department exchange course",
        "advance placement credit",
        "this course consists of selected topics, which may vary from year to year depending on the interests and availability of staff."
    }:
        return False

    if re.fullmatch(r"(faculty )?exchange course( - yus \(1 unit\))?", normalized):
        return False

    return True


def save_cleaned_csv(records, output_path):
    """Save filtered records after removing placeholder descriptions."""
    df = pd.DataFrame(records)
    if "description" not in df.columns:
        raise ValueError("Missing 'description' column for cleaning.")

    cleaned_df = df[df["description"].apply(_is_valid_description)].copy()
    cleaned_df.to_csv(output_path, index=False)
    return len(cleaned_df), len(df) - len(cleaned_df)

def main() -> None:
    """Run the full data extraction pipeline"""
    raw_data = fetch_module_data(url)
    extracted_data = extract_fields(raw_data)
    save_to_csv(extracted_data, output_path)
    cleaned_count, removed_count = save_cleaned_csv(extracted_data, cleaned_output_path)

    print(f"Fetched {len(raw_data)} modules")
    print(f"Saved {len(extracted_data)} records to {output_path}")
    print(
        f"Saved {cleaned_count} cleaned records to {cleaned_output_path} "
        f"(removed {removed_count})"
    )

if __name__ == "__main__":
    main()
