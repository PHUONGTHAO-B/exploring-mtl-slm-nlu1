"""Logger + history (CSV/JSON) writer."""
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


def build_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
    return logger


class HistoryWriter:
    """Append-only CSV+JSON writer for per-epoch metrics."""

    def __init__(self, out_dir: str):
        self.csv_path = Path(out_dir) / "history.csv"
        self.json_path = Path(out_dir) / "history.json"
        self.records: List[Dict[str, Any]] = []
        self._fields: List[str] = []
        if self.json_path.exists():
            try:
                self.records = json.loads(self.json_path.read_text(encoding="utf-8"))
                if self.records:
                    self._fields = list(self.records[0].keys())
            except Exception:
                self.records = []

    def append(self, record: Dict[str, Any]) -> None:
        self.records.append(record)
        for k in record.keys():
            if k not in self._fields:
                self._fields.append(k)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fields); w.writeheader()
            for r in self.records:
                w.writerow({k: r.get(k, "") for k in self._fields})
        self.json_path.write_text(
            json.dumps(self.records, indent=2, default=str), encoding="utf-8"
        )
