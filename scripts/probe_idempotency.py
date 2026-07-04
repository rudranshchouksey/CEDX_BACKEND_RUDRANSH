import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_probe():
    logger.info("Starting probe: Idempotency")
    
    import os
    env = os.environ.copy()
    env["REPLAY_LLM"] = "true"
    env["CASE_ID"] = "CEDX-TESTID"
    
    # Run first time
    logger.info("Executing Run 1...")
    res1 = subprocess.run(
        [sys.executable, "-m", "cedx_pipeline.main"],
        env=env,
        capture_output=True,
        text=True
    )
    if res1.returncode != 0:
        logger.error(f"Run 1 failed: {res1.stderr}")
        sys.exit(1)
        
    audit_path = Path("out/audit.json")
    if not audit_path.exists():
        logger.error("Run 1 did not produce out/audit.json")
        sys.exit(1)
        
    with open(audit_path, "r", encoding="utf-8") as f:
        audit1 = json.load(f)
        
    # Run second time
    logger.info("Executing Run 2...")
    res2 = subprocess.run(
        [sys.executable, "-m", "cedx_pipeline.main"],
        env=env,
        capture_output=True,
        text=True
    )
    if res2.returncode != 0:
        logger.error(f"Run 2 failed: {res2.stderr}")
        sys.exit(1)
        
    with open(audit_path, "r", encoding="utf-8") as f:
        audit2 = json.load(f)
        
    # Compare idempotency criteria
    # Output structures should not be duplicated (e.g. events should not be 2x size)
    events1 = audit1.get("events", [])
    events2 = audit2.get("events", [])
    
    if len(events1) != len(events2):
        logger.error(f"Probe Failed: Event count changed between runs ({len(events1)} vs {len(events2)}). Not idempotent!")
        sys.exit(1)
        
    hash1 = audit1.get("output_package_hash")
    hash2 = audit2.get("output_package_hash")
    
    if hash1 != hash2:
        logger.error("Probe Failed: output_package_hash changed between runs. Output is not deterministic!")
        sys.exit(1)
        
    logger.info("Probe Success: Pipeline runs are fully idempotent and deterministic.")
    sys.exit(0)


if __name__ == "__main__":
    run_probe()
