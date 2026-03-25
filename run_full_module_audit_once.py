from __future__ import annotations

import ast
import importlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE_PARTS = {".venv", "frontend", "node_modules", "__pycache__", ".git"}


def should_include(path: Path) -> bool:
    if path.name in {"execute_all_modules.py", "module_usage_manifest.py", "clean_module_audit.py", "run_full_module_audit_once.py"}:
        return False
    if path.suffix != ".py":
        return False
    return not any(part in EXCLUDE_PARTS for part in path.parts)


def to_module_name(file_path: Path) -> str:
    rel = file_path.relative_to(ROOT).as_posix()
    if rel.endswith("/__init__.py"):
        return rel[: -len("/__init__.py")].replace("/", ".")
    return rel[:-3].replace("/", ".")


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
    files = sorted(path for path in ROOT.rglob("*.py") if should_include(path))
    module_name_map = {path: to_module_name(path) for path in files}
    module_names = set(module_name_map.values())
    referenced_modules = collect_local_references(module_names, files)

    results: list[dict[str, object]] = []

    print(f"Discovered total={len(files)}. Importing all modules in-process...")
    for idx, file_path in enumerate(files, start=1):
        rel = file_path.relative_to(ROOT).as_posix()
        module_name = module_name_map[file_path]
        in_use = module_name in referenced_modules

        if idx % 10 == 0 or idx == 1 or idx == len(files):
            print(f"Progress: {idx}/{len(files)}")

        started_at = time.time()
        try:
            importlib.import_module(module_name)
            status = "success"
            stderr_tail = ""
            stdout_tail = f"imported {module_name}\n"
            returncode = 0
        except Exception as exc:
            status = "failed"
            stderr_tail = str(exc)
            stdout_tail = ""
            returncode = 1

        duration = round(time.time() - started_at, 2)
        results.append(
            {
                "module": rel,
                "module_name": module_name,
                "used_in_import_graph": in_use,
                "status": status,
                "returncode": returncode,
                "duration_seconds": duration,
                "stdout_tail": stdout_tail[-1200:],
                "stderr_tail": stderr_tail[-1200:],
            }
        )

    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if r["status"] == "failed")

    report = {
        "total": len(results),
        "success": success_count,
        "failed": failed_count,
        "timeout": 0,
        "mode": "import-inprocess",
        "timeout_seconds": None,
        "local_modules_discovered": len(files),
        "modules_referenced_in_import_graph": len(referenced_modules),
        "modules_not_referenced": [
            item["module"] for item in results if not bool(item.get("used_in_import_graph"))
        ],
        "results": results,
    }

    (ROOT / "module_execution_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (ROOT / "module_execution_progress.jsonl").open("w", encoding="utf-8") as pf:
        for item in results:
            pf.write(json.dumps(item, ensure_ascii=True) + "\n")

    with (ROOT / "module_execution_report.txt").open("w", encoding="utf-8") as f:
        f.write(
            f"Total: {report['total']} | Success: {report['success']} | Failed: {report['failed']} | Timeout: 0\n\n"
        )
        f.write(
            f"Mode: {report['mode']} | ReferencedModules: {report['modules_referenced_in_import_graph']}/{report['local_modules_discovered']}\n\n"
        )
        for item in results:
            f.write(
                f"[{str(item['status']).upper()}] {item['module']} (rc={item['returncode']}, {item['duration_seconds']}s, used={item['used_in_import_graph']})\n"
            )
            if item["stderr_tail"]:
                f.write("stderr tail:\n")
                f.write(str(item["stderr_tail"]) + "\n")
            f.write("\n")

    write_usage_manifest(sorted(module_names))

    print("\nExecution finished.")
    print(f"Success: {success_count}, Failed: {failed_count}, Timeout: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
