"""
Production Hardening — Load Testing, Caching & Health Checks
=============================================================
Validates the application under realistic production loads,
implements caching for inference, and verifies Docker deployment.

Usage:
    python production_hardening.py --all
    python production_hardening.py --load-test
    python production_hardening.py --cache-demo
    python production_hardening.py --docker-check
"""

import argparse
import json
import os
import sys
import time
import hashlib
import io
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ------------------------------------------------------------------ #
#  1. LRU Inference Cache
# ------------------------------------------------------------------ #

class InferenceCache:
    """
    Thread-safe LRU cache for model predictions.
    Caches based on image content hash to avoid redundant inference calls.
    """

    def __init__(self, max_size: int = 512):
        self.max_size = max_size
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _hash_image(image_bytes: bytes) -> str:
        return hashlib.sha256(image_bytes).hexdigest()

    def get(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        key = self._hash_image(image_bytes)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, image_bytes: bytes, result: Dict[str, Any]) -> None:
        key = self._hash_image(image_bytes)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = result

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        return {
            "cache_size": self.size,
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{self.hit_rate:.2%}",
        }


# ------------------------------------------------------------------ #
#  2. Load Testing
# ------------------------------------------------------------------ #

class LoadTester:
    """
    Simulates concurrent API requests to stress-test the inference server.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        n_workers: int = 10,
        n_requests: int = 100,
    ):
        self.base_url = base_url
        self.n_workers = n_workers
        self.n_requests = n_requests

    def _create_test_image(self) -> bytes:
        img = Image.new("RGB", (224, 224), color=(
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        ))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def _send_request(self, idx: int) -> Dict[str, Any]:
        """Simulate a single prediction request."""
        start = time.perf_counter()
        try:
            import requests
            img_bytes = self._create_test_image()
            files = {"file": ("test.png", io.BytesIO(img_bytes), "image/png")}
            resp = requests.post(
                f"{self.base_url}/api/predict/",
                files=files,
                timeout=30,
            )
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "request_id": idx,
                "status_code": resp.status_code,
                "latency_ms": elapsed,
                "success": resp.status_code == 200,
            }
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return {
                "request_id": idx,
                "status_code": 0,
                "latency_ms": elapsed,
                "success": False,
                "error": str(e),
            }

    def run(self) -> Dict[str, Any]:
        """Execute load test with concurrent workers."""
        print(f"\n{'='*60}")
        print(f"  LOAD TEST: {self.n_requests} requests, {self.n_workers} workers")
        print(f"  Target: {self.base_url}")
        print(f"{'='*60}")

        results: List[Dict] = []
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._send_request, i): i
                for i in range(self.n_requests)
            }
            for future in as_completed(futures):
                results.append(future.result())

        total_time = time.perf_counter() - start_time
        latencies = [r["latency_ms"] for r in results]
        successes = sum(1 for r in results if r["success"])

        summary = {
            "total_requests": self.n_requests,
            "workers": self.n_workers,
            "total_time_s": round(total_time, 2),
            "requests_per_second": round(self.n_requests / total_time, 2),
            "success_count": successes,
            "failure_count": self.n_requests - successes,
            "success_rate": f"{successes / self.n_requests:.2%}",
            "latency_mean_ms": round(np.mean(latencies), 2),
            "latency_p50_ms": round(np.percentile(latencies, 50), 2),
            "latency_p95_ms": round(np.percentile(latencies, 95), 2),
            "latency_p99_ms": round(np.percentile(latencies, 99), 2),
            "latency_max_ms": round(np.max(latencies), 2),
        }

        print(f"\n  RESULTS")
        print(f"  {'-'*50}")
        for k, v in summary.items():
            print(f"  {k:25s}: {v}")

        return summary


# ------------------------------------------------------------------ #
#  3. Docker Deployment Check
# ------------------------------------------------------------------ #

class DockerChecker:
    """Validates Docker and docker-compose configuration."""

    PROJECT_DIR = Path(__file__).resolve().parent

    def check_dockerfile(self) -> Dict[str, Any]:
        """Verify Dockerfile exists and is valid."""
        dfile = self.PROJECT_DIR / "Dockerfile"
        result = {"exists": dfile.exists(), "issues": []}
        if dfile.exists():
            content = dfile.read_text()
            if "FROM" not in content:
                result["issues"].append("Missing FROM instruction")
            if "HEALTHCHECK" not in content:
                result["issues"].append("Missing HEALTHCHECK instruction")
            if "EXPOSE" not in content:
                result["issues"].append("Missing EXPOSE instruction")
            result["lines"] = len(content.splitlines())
        return result

    def check_compose(self) -> Dict[str, Any]:
        """Verify docker-compose.yml exists and defines expected services."""
        cfile = self.PROJECT_DIR / "docker-compose.yml"
        result = {"exists": cfile.exists(), "issues": []}
        if cfile.exists():
            content = cfile.read_text()
            for svc in ["api", "frontend", "neo4j"]:
                if svc not in content:
                    result["issues"].append(f"Missing service: {svc}")
            result["lines"] = len(content.splitlines())
        return result

    def check_requirements(self) -> Dict[str, Any]:
        """Verify requirements.txt is comprehensive."""
        rfile = self.PROJECT_DIR / "requirements.txt"
        result = {"exists": rfile.exists(), "issues": []}
        if rfile.exists():
            content = rfile.read_text()
            essential = ["torch", "fastapi", "uvicorn", "numpy", "Pillow"]
            for pkg in essential:
                if pkg.lower() not in content.lower():
                    result["issues"].append(f"Missing package: {pkg}")
            result["packages"] = len([l for l in content.splitlines() if l.strip() and not l.startswith("#")])
        return result

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all deployment checks."""
        print(f"\n{'='*60}")
        print("  DOCKER DEPLOYMENT CHECK")
        print(f"{'='*60}")

        checks = {
            "dockerfile": self.check_dockerfile(),
            "docker_compose": self.check_compose(),
            "requirements": self.check_requirements(),
        }

        all_issues = []
        for name, result in checks.items():
            status = "OK" if not result.get("issues") else "WARN"
            print(f"  {name:20s}: {status}")
            if result.get("issues"):
                for issue in result["issues"]:
                    print(f"    - {issue}")
                    all_issues.append(f"{name}: {issue}")

        checks["summary"] = {
            "total_issues": len(all_issues),
            "status": "PASS" if len(all_issues) == 0 else "WARNINGS",
        }
        return checks


# ------------------------------------------------------------------ #
#  4. Health Check Verification
# ------------------------------------------------------------------ #

def check_api_health(base_url: str = "http://127.0.0.1:8001") -> Dict[str, Any]:
    """Check if API server is responding correctly."""
    endpoints = [
        ("/", "root"),
        ("/health", "health"),
        ("/api/models", "models"),
    ]
    results = {}
    for path, name in endpoints:
        try:
            import requests
            resp = requests.get(f"{base_url}{path}", timeout=5)
            results[name] = {
                "status_code": resp.status_code,
                "ok": resp.status_code == 200,
                "response_time_ms": resp.elapsed.total_seconds() * 1000,
            }
        except Exception as e:
            results[name] = {"ok": False, "error": str(e)}
    return results


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Production Hardening")
    parser.add_argument("--all", action="store_true", help="Run all checks")
    parser.add_argument("--load-test", action="store_true", help="Run load test")
    parser.add_argument("--cache-demo", action="store_true", help="Demo inference cache")
    parser.add_argument("--docker-check", action="store_true", help="Check Docker files")
    parser.add_argument("--health", action="store_true", help="Check API health")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="API base URL")
    parser.add_argument("--workers", type=int, default=10, help="Load test workers")
    parser.add_argument("--requests", type=int, default=100, help="Load test requests")
    args = parser.parse_args()

    run_all = args.all or not any([args.load_test, args.cache_demo, args.docker_check, args.health])

    print("=" * 60)
    print("  PRODUCTION HARDENING SUITE")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    # 1. Docker check
    if run_all or args.docker_check:
        checker = DockerChecker()
        results["docker"] = checker.run_all_checks()

    # 2. Cache demo
    if run_all or args.cache_demo:
        print(f"\n{'='*60}")
        print("  INFERENCE CACHE DEMO")
        print(f"{'='*60}")
        cache = InferenceCache(max_size=100)
        # Simulate cache usage
        test_images = [Image.new("RGB", (224, 224), color=(i*30, i*20, i*10)) for i in range(10)]
        for i, img in enumerate(test_images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            img_bytes = buf.getvalue()

            # First access: miss
            cached = cache.get(img_bytes)
            assert cached is None
            cache.put(img_bytes, {"class": "genuine", "confidence": 0.95 - i * 0.05})

            # Second access: hit
            cached = cache.get(img_bytes)
            assert cached is not None

        print(f"  {json.dumps(cache.stats(), indent=2)}")
        results["cache"] = cache.stats()

    # 3. Health check
    if run_all or args.health:
        print(f"\n{'='*60}")
        print("  API HEALTH CHECK")
        print(f"{'='*60}")
        health = check_api_health(args.url)
        for name, info in health.items():
            status = "OK" if info.get("ok") else "FAIL"
            print(f"  {name:15s}: {status}")
        results["health"] = health

    # 4. Load test
    if args.load_test:
        tester = LoadTester(
            base_url=args.url,
            n_workers=args.workers,
            n_requests=args.requests,
        )
        results["load_test"] = tester.run()

    # Save results
    output = Path("results/production_hardening.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ Results saved to: {output}")


if __name__ == "__main__":
    main()
