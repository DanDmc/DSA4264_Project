import requests
import pandas as pd

url = "http://api.nusmods.com/v2/2025-2026/moduleInfo.json"
output_path = "modules.csv"

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

def main() -> None:
    """Run the full data extraction pipeline"""
    raw_data = fetch_module_data(url)
    extracted_data = extract_fields(raw_data)
    save_to_csv(extracted_data, output_path)

    print(f"Fetched {len(raw_data)} modules")
    print(f"Saved {len(extracted_data)} records to {output_path}")

if __name__ == "__main__":
    main()