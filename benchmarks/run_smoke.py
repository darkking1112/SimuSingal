"""Small project-specific baseline; not a contractual performance benchmark."""
import argparse
import json
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", choices=["analysis", "simulation"])
    args = parser.parse_args()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as folder:
        request = {"workspace": folder}
        if args.project == "analysis":
            from signal_analysis.tasks import run_job
            asset = run_job({**request, "action": "demo", "count": 8192})
            result = run_job({**request, "action": "analyze", "asset_id": asset["id"]})
        else:
            from communication_sim.tasks import run_job
            result = run_job({**request, "action": "simulate", "scenario": {"messages": 12}})
    print(json.dumps({"project": args.project, "wall_s": time.perf_counter() - started,
                      "summary": result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
