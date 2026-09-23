"""Task 4 final open-set evaluation from the saved cache (manual §4 step 6). CPU is enough.

  python -m task4.evaluate_osr

Tables -> report/tables/task4_*.csv ; figures -> report/figures/task4/ ; all numbers -> task4/results/final/metrics.json
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common.io import load_json, save_json
from common.paths import REPO_ROOT, TABLE_DIR, task_dir
from common.plotting import savefig, setup_style
from task4.data.cifar10 import CLASSES
from task4.data.cifar100_unknowns import FAR, LOCK, NEAR, verify_lock
from task4.evaluation.failure_analysis import absorption, accepted_unknowns, most_confident_per_class
from task4.evaluation.metrics import accuracy, auroc, roc
from task4.evaluation.thresholds import accept_rate, val_threshold
from task4.extract_outputs import CACHE, RUNS
from task4.scores import Mahalanobis, energy, mls, msp, proser_placeholder

PARTS = ("val", "test", "unknown")
SCORE_NAMES = {"msp": "MSP", "mls": "MLS", "energy": "Energy", "mahalanobis": "Mahalanobis",
               "placeholder": "PROSER placeholder"}
FIG = "task4"


def scores_for(c, name, maha=None):
    if name == "mahalanobis":
        return {p: maha(c[f"{p}_feats"]) for p in PARTS}
    if name == "placeholder":
        return {p: proser_placeholder(c[f"{p}_logits"], c[f"{p}_dummy"]) for p in PARTS}
    fn = {"msp": msp, "mls": mls, "energy": energy}[name]
    return {p: fn(c[f"{p}_logits"]) for p in PARTS}


def osr_metrics(u, c):
    g = c["unknown_group"]
    near, far, unk = u["unknown"][g == "near"], u["unknown"][g == "far"], u["unknown"]
    tau = val_threshold(u["val"])
    m = {"CSA": accuracy(c["test_logits"], c["test_y"]),
         "AUROC_near": auroc(u["test"], near), "AUROC_far": auroc(u["test"], far), "AUROC_all": auroc(u["test"], unk),
         "tau": tau, "val_accept": accept_rate(u["val"], tau), "test_accept": accept_rate(u["test"], tau),
         "reject_near": 1 - accept_rate(near, tau), "reject_far": 1 - accept_rate(far, tau),
         "reject_all": 1 - accept_rate(unk, tau)}
    m.update({f"FPR95_{k}": 1 - m[f"reject_{k}"] for k in ("near", "far", "all")})
    return m


def main():
    lk = verify_lock()
    main_p = lk["proser_main"]
    other_p = "proser" if main_p == "proser_cleanbn" else "proser_cleanbn"
    C = {r: dict(np.load(CACHE / f"{r}.npz", allow_pickle=False)) for r in RUNS}
    van = C["vanilla"]
    maha = Mahalanobis().fit(van["train_feats"], van["train_y"])
    grp, cls = van["unknown_group"], van["unknown_class"]
    setup_style()
    res = {"lock": lk, "proser_main": main_p}

    # ---- Table 1: post-hoc scores on the frozen Vanilla model
    U = {s: scores_for(van, s, maha) for s in ("msp", "mls", "energy", "mahalanobis")}
    t1 = pd.DataFrame([{"score": SCORE_NAMES[s], **osr_metrics(U[s], van)} for s in U])
    t1.round(4).to_csv(TABLE_DIR / "task4_posthoc_scores.csv", index=False)
    res["posthoc"] = t1.to_dict("records")

    # ---- Table 2: trained models (MLS common score + PROSER placeholder score)
    label = {"proser": "PROSER (original BN)", "proser_cleanbn": "PROSER (clean BN)"}
    combos = [("Vanilla", "vanilla", "mls"), ("GCSC", "gcsc", "mls"),
              (label[main_p] + " [main]", main_p, "mls"), (label[main_p] + " [main]", main_p, "placeholder"),
              (label[other_p], other_p, "mls"), (label[other_p], other_p, "placeholder")]
    UM, rows = {}, []
    for name, run, s in combos:
        UM[(run, s)] = u = scores_for(C[run], s)
        rows.append({"model": name, "run": run, "checkpoint_epoch": int(C[run]["epoch"]), "score": SCORE_NAMES[s],
                     **osr_metrics(u, C[run])})
    t2 = pd.DataFrame(rows)
    t2.round(4).to_csv(TABLE_DIR / "task4_model_comparison.csv", index=False)
    res["models"] = t2.to_dict("records")

    # ---- per-unknown-class acceptance for every evaluated (model, score)
    allc = [("vanilla", s) for s in U] + [k for k in UM if k != ("vanilla", "mls")]
    Uall = {**{("vanilla", s): U[s] for s in U}, **UM}
    pc = pd.DataFrame({"unknown_class": list(NEAR) + list(FAR), "group": ["near"] * 8 + ["far"] * 8})
    for run, s in allc:
        u, tau = Uall[(run, s)], val_threshold(Uall[(run, s)]["val"])
        pc[f"{run}:{s}"] = [accept_rate(u["unknown"][cls == k], tau) for k in pc.unknown_class]
    pc.round(4).to_csv(TABLE_DIR / "task4_per_class_acceptance.csv", index=False)

    # ---- absorption (which CIFAR-10 label absorbs accepted unknowns), MLS for the three models
    absr = []
    for run in ("vanilla", "gcsc", main_p):
        u = Uall[(run, "mls")]
        acc = accepted_unknowns(u["unknown"], C[run]["unknown_logits"].argmax(1), grp, cls,
                                van["unknown_hf_index"], val_threshold(u["val"]))
        a = absorption(acc, list(NEAR) + list(FAR))
        a.insert(0, "run", run)
        absr.append(a.reset_index())
    pd.concat(absr).to_csv(TABLE_DIR / "task4_absorption_mls.csv", index=False)

    # ---- agreement between the four post-hoc scores (Vanilla)
    names = list(U)
    cat = {s: np.r_[U[s]["test"], U[s]["unknown"]] for s in names}
    tau = {s: val_threshold(U[s]["val"]) for s in names}
    ag = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = {"pair": f"{SCORE_NAMES[a]} vs {SCORE_NAMES[b]}", "spearman_all": spearmanr(cat[a], cat[b])[0]}
            for part, sel in (("known_test", None), ("near", grp == "near"), ("far", grp == "far")):
                ua, ub = (U[a]["test"], U[b]["test"]) if sel is None else (U[a]["unknown"][sel], U[b]["unknown"][sel])
                r[f"spearman_{part}"] = spearmanr(ua, ub)[0]
                r[f"decision_disagree_{part}"] = float(((ua <= tau[a]) != (ub <= tau[b])).mean())
            ag.append(r)
    pd.DataFrame(ag).round(4).to_csv(TABLE_DIR / "task4_score_agreement.csv", index=False)

    # ---- failure analysis: Vanilla MLS
    u = U["mls"]
    t_mls = val_threshold(u["val"])
    acc = accepted_unknowns(u["unknown"], van["unknown_logits"].argmax(1), grp, cls, van["unknown_hf_index"], t_mls)
    acc.round(4).to_csv(task_dir("task4", "results", "final") / "vanilla_mls_accepted_unknowns.csv", index=False)
    fails = most_confident_per_class(acc)
    fails.round(4).to_csv(TABLE_DIR / "task4_failures_vanilla_mls.csv", index=False)
    pos = {h: i for i, h in enumerate(van["unknown_hf_index"])}
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.8))
    for r, g in enumerate(("near", "far")):
        sub = fails[fails.group == g].head(8)
        for k, ax in enumerate(axes[r]):
            ax.axis("off")
            if k < len(sub):
                e = sub.iloc[k]
                ax.imshow(van["unknown_images"][pos[e.hf_test_index]], interpolation="nearest")
                ax.set_title(f"{e.unknown_class}\n→ {e.predicted}\nu={e.score:.2f}", fontsize=7)
        axes[r][0].text(-0.35, 0.5, f"{g}\nτ={t_mls:.2f}", transform=axes[r][0].transAxes, ha="right", va="center")
    fig.suptitle("Vanilla + MLS: most confidently accepted unknown per class (accepted iff u ≤ τ)", fontsize=9)
    savefig(fig, "task4_failures_vanilla_mls", FIG)

    # ---- figure: MSP / MLS / Mahalanobis distributions + ROC
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for j, s in enumerate(("msp", "mls", "mahalanobis")):
        uu, ax = U[s], axes[0][j]
        data = {"CIFAR-10 test": uu["test"], "near": uu["unknown"][grp == "near"], "far": uu["unknown"][grp == "far"]}
        bins = np.histogram_bin_edges(np.concatenate(list(data.values())), 60)
        for (k, v), col in zip(data.items(), ("C0", "C1", "C2")):
            ax.hist(v, bins=bins, density=True, histtype="step", lw=1.3, color=col, label=k)
        ax.axvline(tau[s], color="k", ls="--", lw=1, label="τ (95% val)")
        ax.set_yscale("log"); ax.set_title(f"{SCORE_NAMES[s]} unknownness"); ax.set_xlabel("u(x)")
        ax.legend(fontsize=7)
        ax = axes[1][j]
        for k, sel, col in (("near", grp == "near", "C1"), ("far", grp == "far", "C2")):
            f, t = roc(uu["test"], uu["unknown"][sel])
            ax.plot(f, t, color=col, label=f"{k} (AUROC {auroc(uu['test'], uu['unknown'][sel]):.3f})")
        ax.plot([0, 1], [0, 1], color="0.7", lw=0.8)
        ax.set_xlabel("FPR (known rejected)"); ax.set_ylabel("TPR (unknown rejected)"); ax.legend(fontsize=7)
    fig.tight_layout()
    savefig(fig, "task4_score_distributions_roc", FIG)

    # ---- figure: training curves
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    R = REPO_ROOT / "task4/results"
    for run, col in (("vanilla", "C0"), ("gcsc", "C1")):
        d = pd.read_csv(R / run / "train_epochs.csv")
        axes[0].plot(d.epoch, d.val_acc, color=col, label=f"{run} val acc")
        axes[0].plot(d.epoch, d.train_acc, color=col, ls=":", label=f"{run} train acc")
    axes[0].set_ylim(0.8, 1.0); axes[0].set_xlabel("epoch"); axes[0].legend(fontsize=7)
    for run, ls in (("proser", "--"), ("proser_cleanbn", "-")):
        d = pd.read_csv(R / run / "train_epochs.csv")
        axes[1].plot(d.epoch, d.val_acc, ls=ls, color="C2", label=f"{label[run]} val acc")
        axes[1].plot(d.epoch, d.train_mix_to_dummy, ls=ls, color="C3", label=f"{label[run]} mix→dummy")
        for k, col in (("train_l_known", "C0"), ("train_l_classifier_ph", "C4"), ("train_l_data_ph", "C1")):
            axes[2].plot(d.epoch, d[k], ls=ls, color=col, label=f"{k[6:]} ({'orig' if run == 'proser' else 'clean'})")
    axes[1].set_xlabel("PROSER fine-tune epoch"); axes[1].legend(fontsize=6)
    axes[2].set_yscale("log"); axes[2].set_xlabel("PROSER fine-tune epoch"); axes[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    savefig(fig, "task4_training_curves", FIG)

    save_json(res, task_dir("task4", "results", "final") / "metrics.json")
    pd.set_option("display.width", 200)
    cols = ["CSA", "AUROC_near", "AUROC_far", "AUROC_all", "test_accept", "reject_near", "reject_far"]
    print(t1[["score"] + cols].round(4).to_string(index=False))
    print(t2[["model", "score", "checkpoint_epoch"] + cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
