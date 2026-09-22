"""Per-domain metrics for Task 3 tables: each source val domain, mean, worst
(source-side; manual §3 Step 4) and, in the final Sketch evaluation only, target
metrics and the change relative to ERM. Wraps the Task 2 ``domain_report``."""
from shared.pacs_protocol import SOURCES
from task2.evaluation.metrics import domain_report  # noqa: F401


def source_row(rep: dict) -> dict:
    row = {}
    for d in SOURCES:
        row[f"val_{d}_acc"] = rep[d]["accuracy"]
        row[f"val_{d}_f1"] = rep[d]["macro_f1"]
    row.update(val_mean_acc=rep["mean_accuracy"], val_mean_f1=rep["mean_macro_f1"],
               val_worst_acc=rep["worst_accuracy"], val_worst_f1=rep["worst_macro_f1"],
               val_worst_domain=rep["worst_domain_by_f1"])
    return row
