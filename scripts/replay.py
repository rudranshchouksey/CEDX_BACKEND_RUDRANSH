import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Reconstruct state history for a record.")
    parser.add_argument("--id", required=True, help="Record ID to replay")
    args = parser.parse_args()

    audit_path = Path("out/audit.json")
    if not audit_path.exists():
        print("Error: out/audit.json not found. Run 'make demo' first.", file=sys.stderr)
        sys.exit(1)

    with open(audit_path, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    events = audit_data.get("events", [])
    record_id = args.id

    record_events = [
        e for e in events 
        if e.get("details", {}).get("record_id") == record_id
    ]

    if not record_events:
        print(f"No events found for record ID: {record_id}")
        sys.exit(1)

    print(f"=== State History for Record {record_id} ===")
    for event in record_events:
        seq = event.get("seq")
        e_type = event.get("event_type")
        ts = event.get("timestamp")
        details = event.get("details", {})
        
        if e_type == "STATE_TRANSITION":
            old = details.get("old_state")
            new = details.get("new_state")
            print(f"[{ts}] (seq {seq}) Transition: {old} -> {new}")
        else:
            print(f"[{ts}] (seq {seq}) {e_type}: {details}")


if __name__ == "__main__":
    main()
