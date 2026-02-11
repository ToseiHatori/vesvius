#!/home/ben/.local/share/pipx/venvs/kaggle/bin/python
"""Kaggle Submission 実行時間計測スクリプト.

最新のsubmissionをポーリングし、実行時間とLBスコアを表示する。
参考: https://zenn.dev/currypurin/scraps/47d5f84a0ca89d
"""

import argparse
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def extract_notebook_name(submission) -> str:
    """submissionのURLからノートブック名を抽出する。

    URLは通常 /code/user/notebook-slug 形式。
    取得できない場合はfile_nameやdescriptionにフォールバック。
    """
    url = getattr(submission, "url", "") or ""
    if url:
        # /code/user/notebook-slug or /code/user/notebook-slug/...
        m = re.search(r"/code/[^/]+/([^/?]+)", url)
        if m:
            return m.group(1)

    file_name = getattr(submission, "file_name", "") or ""
    if file_name:
        return Path(file_name).stem

    description = getattr(submission, "description", "") or ""
    if description:
        # 安全なファイル名にする
        return re.sub(r"[^\w\-]", "_", description)[:60]

    return "unknown"


def setup_logging(log_dir: str, submission) -> logging.Logger:
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    notebook_name = extract_notebook_name(submission)
    ref = submission.ref
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_path / f"{notebook_name}_ref{ref}_{timestamp}.log"

    logger = logging.getLogger("monitor_submission")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"Log file: {log_file}")
    return logger


def get_status(submission) -> str:
    """statusを文字列として取得する。enum/文字列どちらにも対応。"""
    status = submission.status
    if hasattr(status, "name"):
        return status.name.lower()  # PENDING -> pending, COMPLETE -> complete
    return str(status).lower()


def monitor(competition: str, interval: int, log_dir: str) -> None:
    api = KaggleApi()
    api.authenticate()

    submissions = api.competition_submissions(competition)
    if not submissions:
        print("No submissions found.")
        return

    latest = submissions[0]
    ref = latest.ref
    submit_time = latest.date

    logger = setup_logging(log_dir, latest)
    logger.info(f"Competition: {competition}")
    logger.info(f"Notebook: {extract_notebook_name(latest)}")
    logger.info(f"Tracking submission: ref={ref}")
    logger.info(f"Submit time: {submit_time}")

    while True:
        submissions = api.competition_submissions(competition)
        current = next((s for s in submissions if s.ref == ref), None)
        if current is None:
            logger.info(f"Submission ref={ref} not found. Stopping.")
            return

        # submit_time is timezone-naive, assume it's UTC
        if submit_time.tzinfo is None:
            submit_time = submit_time.replace(tzinfo=timezone.utc)
        elapsed = datetime.now(timezone.utc) - submit_time
        elapsed_min = elapsed.total_seconds() / 60

        status = get_status(current)
        logger.info(f"Status: {status} | Elapsed: {elapsed_min:.1f} min")

        if status == "complete":
            score = current.public_score
            logger.info(f"Completed in {elapsed_min:.1f} min | LB Score: {score}")
            return

        if status == "error":
            error_desc = getattr(current, "error_description", "") or ""
            logger.info(f"Submission failed with error. {error_desc}")
            return

        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor Kaggle submission status")
    parser.add_argument(
        "--competition",
        default="vesuvius-challenge-surface-detection",
        help="Competition slug (default: vesuvius-challenge-surface-detection)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Polling interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Log output directory (default: logs)",
    )
    args = parser.parse_args()
    monitor(args.competition, args.interval, args.log_dir)


if __name__ == "__main__":
    main()
