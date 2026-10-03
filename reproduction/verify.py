"""Fail closed on incomplete or different replay inputs; does not calculate scores."""

import argparse
import json
from pathlib import Path

from .integrity import check_count, inventory, sha256


def verify(
    gt: Path,
    pred: Path,
    config: Path,
    expected_count: int,
    expected_gt: str,
    expected_pred: str,
    expected_config: str,
) -> dict:
    ground_truth, predictions = inventory(gt, ".md"), inventory(pred, ".md")
    for item, expected, label in (
        (ground_truth, expected_gt, "GT"),
        (predictions, expected_pred, "prediction"),
    ):
        check_count(item, expected_count)
        if item["collection_sha256"] != expected:
            raise ValueError(f"{label} fingerprint mismatch")
    if ground_truth["files"].keys() != predictions["files"].keys():
        missing = sorted(ground_truth["files"].keys() - predictions["files"].keys())
        extra = sorted(predictions["files"].keys() - ground_truth["files"].keys())
        raise ValueError(f"page names differ: missing={missing}, extra={extra}")
    config_hash = sha256(config)
    if config_hash != expected_config:
        raise ValueError("config fingerprint mismatch")
    return {
        "pages": expected_count,
        "gt_sha256": expected_gt,
        "prediction_sha256": expected_pred,
        "config_sha256": config_hash,
        "input_verification": "passed",
        "scoring_verification": "not_run",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=1400)
    parser.add_argument("--expected-gt-sha256", required=True)
    parser.add_argument("--expected-pred-sha256", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    args = parser.parse_args()
    try:
        report = verify(
            args.gt,
            args.pred,
            args.config,
            args.expected_count,
            args.expected_gt_sha256,
            args.expected_pred_sha256,
            args.expected_config_sha256,
        )
    except (ValueError, OSError) as error:
        parser.exit(1, f"verification failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
