"""Minimal CSV logger for training curves (required evidence in Tasks 2-4)."""
import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, path, resume: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not resume and self.path.exists():
            self.path.unlink()
        self._fields = None
        if self.path.exists():
            with open(self.path) as f:
                header = f.readline().strip()
                self._fields = header.split(",") if header else None

    def log(self, row: dict) -> None:
        if self._fields is None:
            self._fields = list(row.keys())
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._fields, extrasaction="ignore").writerow(row)
