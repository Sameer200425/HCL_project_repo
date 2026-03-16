"""Wrapper to run kfold_ensemble.py and capture output to a file."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "kfold_ensemble.py"],
    capture_output=True,
    text=True,
    cwd=".",
)

with open("results/kfold_log_full.txt", "w") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout)
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr)
    f.write(f"\n=== RETURN CODE: {result.returncode} ===\n")

print(f"Return code: {result.returncode}")
if result.returncode != 0:
    print("STDERR (last 2000 chars):")
    print(result.stderr[-2000:])
else:
    # Print last 3000 chars of stdout
    print("STDOUT (last 3000 chars):")
    print(result.stdout[-3000:])
