"""
Quick dry-run test — verifies data loading and text preparation work.
No model download, runs in seconds.

Usage (from repo root):
    python src/embedding/test_dry_run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from config import MODULES_FILE, JOBS_FILE
from embed import prepare_module_texts, prepare_job_texts

print("Loading modules...")
modules_df = pd.read_csv(MODULES_FILE)
print(f"  → {len(modules_df)} rows")
print(f"  → Columns: {list(modules_df.columns)}")

print("\nLoading jobs...")
jobs_df = pd.read_csv(JOBS_FILE)
print(f"  → {len(jobs_df)} rows")
print(f"  → Columns: {list(jobs_df.columns)[:10]}...")

print("\nPreparing module texts...")
mt = prepare_module_texts(modules_df)
print(f"  → {len(mt)} texts prepared")
print(f"  → Sample: {mt[0][:200]}...")

print("\nPreparing job texts...")
jt = prepare_job_texts(jobs_df)
print(f"  → {len(jt)} texts prepared")
print(f"  → Sample: {jt[0][:200]}...")

print("\n✓ All good — data loads and text prep works. Safe to run embed.py.")
