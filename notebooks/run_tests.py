# Databricks notebook source

# COMMAND ----------

# MAGIC %pip install "insurance-gas @ git+https://github.com/burning-cost/insurance-gas.git" pytest

# COMMAND ----------

import subprocess, sys, os, uuid

tmpdir = f"/tmp/insurance-gas-{uuid.uuid4().hex[:8]}"

result = subprocess.run(
    ["git", "clone", "-q", "https://github.com/burning-cost/insurance-gas.git", tmpdir],
    capture_output=True, text=True
)
if result.returncode != 0:
    raise Exception(f"Clone failed: {result.stderr}")
print(f"Cloned to {tmpdir}")

# COMMAND ----------

import subprocess, sys

result = subprocess.run(
    [sys.executable, "-m", "pytest", f"{tmpdir}/tests/", "--tb=short", "-v",
     f"--junit-xml={tmpdir}/test_results.xml"],
    capture_output=True, text=True, cwd=tmpdir
)

combined = result.stdout + "\n" + result.stderr
# Write to a file that we can retrieve
with open(f"{tmpdir}/pytest_output.txt", "w") as f:
    f.write(combined)

# Print last section (failures)
print(combined[-15000:] if len(combined) > 15000 else combined)

# Return output as notebook result for SDK retrieval
dbutils.notebook.exit(combined[-5000:] if len(combined) > 5000 else combined)
