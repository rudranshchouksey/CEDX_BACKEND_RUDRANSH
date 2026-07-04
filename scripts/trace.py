import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Trace a record through the audit log.")
    parser.add_argument("--id", required=True, help="Record ID to trace")
    args = parser.parse_args()

    audit_path = Path("out/audit.json")
    if not audit_path.exists():
        print("Error: out/audit.json not found. Run 'make demo' first.", file=sys.stderr)
        sys.exit(1)

    with open(audit_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    record_id = args.id
    traces = audit_data.get("agent_trace", {}).get(record_id)

    if not traces:
        print(f"No trace found for record ID: {record_id}")
        sys.exit(1)

    print(f"=== Trace for Record {record_id} ===")
    total_cost = 0.0
    total_in = 0
    total_out = 0

    for i, span in enumerate(traces, 1):
        agent = span.get("agent")
        model = span.get("model")
        t_in = span.get("tokens_in", 0)
        t_out = span.get("tokens_out", 0)
        cost = span.get("cost_usd", 0.0)
        verdict = span.get("verdict", "N/A")

        total_cost += cost
        total_in += t_in
        total_out += t_out

        print(f"Step {i}: Agent [{agent}] using Model [{model}]")
        print(f"  Tokens: {t_in} in / {t_out} out")
        print(f"  Cost: ${cost:.6f}")
        if verdict is not None and verdict != "N/A":
            print(f"  Verdict: {verdict}")
        print()

    print("=== Summary ===")
    print(f"Total Tokens: {total_in + total_out}")
    print(f"Total Cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
