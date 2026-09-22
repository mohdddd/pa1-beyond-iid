"""Before/after table for the alignment-branch stabilisation.

Compares the archived unstabilised runs (task2/results/unstable/tables/) with the
current stabilised results (report/tables/), producing
report/tables/task2_stabilisation_comparison.csv.

    python -m task2.evaluation.compare_stabilisation
"""
import pandas as pd

from common.paths import REPO_ROOT, TABLE_DIR

OLD = REPO_ROOT / "task2" / "results" / "unstable" / "tables"
COLS = ["val_mean_f1", "val_worst_f1", "target_acc", "target_f1", "domain_separability",
        "feat_norm_source", "feat_norm_target"]


def main():
    old = pd.read_csv(OLD / "task2_main_comparison.csv").set_index("method")
    new = pd.read_csv(TABLE_DIR / "task2_main_comparison.csv").set_index("method")
    rows = []
    for m in new.index:
        if m not in old.index:
            continue
        r = {"method": m}
        for c in COLS:
            r[f"{c}_before"] = old.loc[m, c]
            r[f"{c}_after"] = new.loc[m, c]
        rows.append(r)
    main_cmp = pd.DataFrame(rows).round(4)
    olds = pd.read_csv(OLD / "task2_controlled_study.csv").set_index("lambda_mmd")
    news = pd.read_csv(TABLE_DIR / "task2_controlled_study.csv").set_index("lambda_mmd")
    srows = []
    for lam in news.index:
        if lam not in olds.index:
            continue
        srows.append({"lambda_mmd": lam,
                      **{f"{c}_before": olds.loc[lam, c] for c in ("val_mean_macro_f1", "target_accuracy", "domain_separability")},
                      **{f"{c}_after": news.loc[lam, c] for c in ("val_mean_macro_f1", "target_accuracy", "domain_separability")}})
    study_cmp = pd.DataFrame(srows).round(4)
    out = TABLE_DIR / "task2_stabilisation_comparison.csv"
    main_cmp.to_csv(out, index=False)
    study_cmp.to_csv(TABLE_DIR / "task2_stabilisation_comparison_study.csv", index=False)
    print(main_cmp.to_string(index=False)); print(); print(study_cmp.to_string(index=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
