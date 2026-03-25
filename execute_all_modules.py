from __future__ import annotations

import ast
import json
import subprocess
import sys
import time
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_TIMEOUT_SECONDS = 20

EXCLUDE_PARTS = {
    ".venv",
    "frontend",
    "node_modules",
    "__pycache__",
    ".git",
}


def should_include(path: Path) -> bool:
    if path.name in {"execute_all_modules.py", "module_usage_manifest.py", "clean_module_audit.py"}:
        return False
    if path.suffix != ".py":
        return False
    return not any(part in EXCLUDE_PARTS for part in path.parts)


def to_module_name(file_path: Path) -> str:
    rel = file_path.relative_to(ROOT).as_posix()
    if rel.endswith("/__init__.py"):
        return rel[: -len("/__init__.py")].replace("/", ".")
    if rel.endswith(".py"):
        return rel[:-3].replace("/", ".")
    return rel.replace("/", ".")


def build_cmd(file_path: Path, module_name: str, mode: str) -> list[str]:
    if mode == "execute":
        return [sys.executable, str(file_path)]
    return [
        sys.executable,
        "-c",
        (
            "import importlib; "
            f"importlib.import_module('{module_name}'); "
            f"print('imported {module_name}')"
        ),
    ]


def collect_local_references(module_names: set[str], files: list[Path]) -> set[str]:
    refs: set[str] = set()
    for path in files:
        try:
            src = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported = alias.name
                    if imported in module_names:
                        refs.add(imported)
                    # Keep parent package references for local packages.
                    for idx in range(imported.count("."), 0, -1):
                        parent = ".".join(imported.split(".")[:idx])
                        if parent in module_names:
                            refs.add(parent)
            elif isinstance(node, ast.ImportFrom):
                if node.level and not node.module:
                    continue
                if node.module:
                    base = node.module
                    if base in module_names:
                        refs.add(base)
                    for alias in node.names:
                        candidate = f"{base}.{alias.name}"
                        if candidate in module_names:
                            refs.add(candidate)

    return refs


def write_usage_manifest(module_names: list[str]) -> Path:
    out = ROOT / "module_usage_manifest.py"
    lines = [
        '"""Auto-generated module usage manifest. Do not edit manually."""',
        "",
        "from __future__ import annotations",
        "",
        "import importlib",
        "from typing import Dict, List",
        "",
        "MODULES: List[str] = [",
    ]
    for name in module_names:
        lines.append(f'    "{name}",')
    lines.extend(
        [
            "]",
            "",
            "def touch_all_modules() -> Dict[str, object]:",
            "    loaded: List[str] = []",
            "    failed: List[dict[str, str]] = []",
            "    for module_name in MODULES:",
            "        try:",
            "            importlib.import_module(module_name)",
            "            loaded.append(module_name)",
            "        except Exception as exc:",
            "            failed.append({'module': module_name, 'error': str(exc)})",
            "    return {'total': len(MODULES), 'loaded': len(loaded), 'failed': failed}",
            "",
            "",
            "if __name__ == '__main__':",
            "    summary = touch_all_modules()",
            "    print(f\"Loaded: {summary['loaded']}/{summary['total']}\")",
            "    if summary['failed']:",
            "        print('Failed modules:')",
            "        for item in summary['failed']:",
            "            print(f\"- {item['module']}: {item['error']}\")",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute/import Python modules with usage reporting")
    parser.add_argument(
        "--mode",
        choices=["import", "execute"],
        default="import",
        help="import = safe module import check (default), execute = run file as script",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="Per-module timeout in seconds",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index in discovered module list")
    parser.add_argument("--limit", type=int, default=0, help="How many modules to execute from start (0=all)")
    parser.add_argument("--append", action="store_true", help="Append progress instead of resetting progress file")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Generate module_usage_manifest.py with all discovered modules",
    )
    args = parser.parse_args()

    files = sorted(path for path in ROOT.rglob("*.py") if should_include(path))
    module_name_map = {path: to_module_name(path) for path in files}
    module_names = set(module_name_map.values())
    referenced_modules = collect_local_references(module_names, files)

    total_files = len(files)

    start = max(args.start, 0)
    if args.limit and args.limit > 0:
        files = files[start : start + args.limit]
    else:
        files = files[start:]

    progress_jsonl = ROOT / "module_execution_progress.jsonl"
    if progress_jsonl.exists() and not args.append:
        progress_jsonl.unlink()

    results: list[dict[str, object]] = []
    if args.append and progress_jsonl.exists():
        for line in progress_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    results.append(item)
            except json.JSONDecodeError:
                continue

    # Keep only the latest record per module when appending multiple batches.
    dedup: dict[str, dict[str, object]] = {}
    for item in results:
        module = item.get("module")
        if isinstance(module, str):
            dedup[module] = item
    if dedup:
        results = list(dedup.values())

    print(f"Discovered total={total_files}. Executing batch size={len(files)} from index={start}.")

    try:
        for idx, file_path in enumerate(files, start=1):
            rel = file_path.relative_to(ROOT).as_posix()
            module_name = module_name_map[file_path]
            in_use = module_name in referenced_modules

            if idx % 10 == 0 or idx == 1 or idx == len(files):
                print(f"Progress: {idx}/{len(files)}")
            started_at = time.time()
            try:
                cmd = build_cmd(file_path, module_name, args.mode)
                env = os.environ.copy()
                env["PYTHONUTF8"] = "1"
                env["PYTHONIOENCODING"] = "utf-8"
                proc = subprocess.run(
                    cmd,
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=args.timeout,
                    env=env,
                )
                duration = round(time.time() - started_at, 2)
                item = {
                    "module": rel,
                    "module_name": module_name,
                    "used_in_import_graph": in_use,
                    "status": "success" if proc.returncode == 0 else "failed",
                    "returncode": proc.returncode,
                    "duration_seconds": duration,
                    "stdout_tail": proc.stdout[-1200:],
                    "stderr_tail": proc.stderr[-1200:],
                }
            except subprocess.TimeoutExpired as exc:
                duration = round(time.time() - started_at, 2)
                item = {
                    "module": rel,
                    "module_name": module_name,
                    "used_in_import_graph": in_use,
                    "status": "timeout",
                    "returncode": None,
                    "duration_seconds": duration,
                    "stdout_tail": (exc.stdout or "")[-1200:] if isinstance(exc.stdout, str) else "",
                    "stderr_tail": (exc.stderr or "")[-1200:] if isinstance(exc.stderr, str) else "",
                }

            # Replace existing entry for a module when a module is re-run.
            existing_idx = next(
                (i for i, rec in enumerate(results) if rec.get("module") == item["module"]),
                -1,
            )
            if existing_idx >= 0:
                results[existing_idx] = item
            else:
                results.append(item)
            with progress_jsonl.open("a", encoding="utf-8") as pf:
                pf.write(json.dumps(item, ensure_ascii=True) + "\n")
    except KeyboardInterrupt:
        print("Interrupted. Writing partial report...")

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")
    timeout_count = sum(1 for r in results if r["status"] == "timeout")

    report = {
        "total": len(results),
        "success": success_count,
        "failed": failed_count,
        "timeout": timeout_count,
        "mode": args.mode,
        "timeout_seconds": args.timeout,
        "local_modules_discovered": total_files,
        "modules_referenced_in_import_graph": len(referenced_modules),
        "modules_not_referenced": [
            item["module"]
            for item in results
            if not bool(item.get("used_in_import_graph"))
        ],
        "results": results,
    }

    out_json = ROOT / "module_execution_report.json"
    out_txt = ROOT / "module_execution_report.txt"

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(
            f"Total: {report['total']} | Success: {report['success']} | "
            f"Failed: {report['failed']} | Timeout: {report['timeout']}\n\n"
        )
        f.write(
            f"Mode: {report['mode']} | TimeoutSeconds: {report['timeout_seconds']} | "
            f"ReferencedModules: {report['modules_referenced_in_import_graph']}/{report['local_modules_discovered']}\n\n"
        )
        for item in results:
            f.write(
                f"[{str(item['status']).upper()}] {item['module']} "
                f"(rc={item['returncode']}, {item['duration_seconds']}s, used={item['used_in_import_graph']})\n"
            )
            if item["stderr_tail"]:
                f.write("stderr tail:\n")
                f.write(str(item["stderr_tail"]) + "\n")
            f.write("\n")

        if report["modules_not_referenced"]:
            f.write("Modules not referenced by local imports:\n")
            for module_path in report["modules_not_referenced"]:
                f.write(f"- {module_path}\n")

    manifest_path = None
    if args.write_manifest:
        manifest_path = write_usage_manifest(sorted(module_names))

    print("\nExecution finished.")
    print(f"Success: {success_count}, Failed: {failed_count}, Timeout: {timeout_count}")
    print(f"Report JSON: {out_json}")
    print(f"Report TXT : {out_txt}")
    if manifest_path:
        print(f"Usage manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
