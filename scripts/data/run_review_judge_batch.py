#!/usr/bin/env python3
"""Submit or collect the PrimeVul review-judge OpenAI batch."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.judge_megavul_probe_fault_openai import (
    collect_batch,
    latest_batch_id,
    retrieve_batch,
    submit_batch,
    write_json,
)


DEFAULT_OUT = Path("data/processed/capability_staging/primevul_megavul_v1/review_judge_inputs_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--batch-id")
    parser.add_argument("--metadata-job", default="primevul_review_judge_v1")
    parser.add_argument("--poll-interval", type=int, default=60)
    args = parser.parse_args()

    batch_id = args.batch_id
    if args.submit:
        manifest = submit_batch(args.out_dir, metadata_job=args.metadata_job)
        batch_id = manifest["batch_id"]
        print(f"SUBMITTED batch_id={batch_id} requests={manifest['num_requests']}", flush=True)

    if args.collect or args.wait:
        batch_id = batch_id or latest_batch_id(args.out_dir)
        if not batch_id:
            raise SystemExit("No --batch-id supplied and no openai_batch_submission.json found.")
        while True:
            batch = retrieve_batch(batch_id)
            print(
                f"BATCH {batch_id} status={batch['status']} counts={batch.get('request_counts')}",
                flush=True,
            )
            if batch["status"] in {"completed", "expired", "cancelled", "failed"}:
                rows = collect_batch(batch_id, args.out_dir, artifact_prefix="review_judge")
                print(f"COLLECTED rows={len(rows)}", flush=True)
                break
            if not args.wait:
                write_json(args.out_dir / "openai_batch_status.json", batch)
                break
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
