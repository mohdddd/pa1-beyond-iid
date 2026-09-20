"""t-SNE / UMAP visualisation of clean vs transformed representations.

One projection is fitted per backbone and intervention on the COMBINED clean and
transformed features, so both conditions live in the same space. Colour = ground
truth class, marker = condition. Coordinates are not comparable across separate
projections.
"""
import numpy as np

from common.plotting import savefig, setup_style
from task1.analysis.feature_similarity import INTERVENTIONS, paired_features
from task1.configs import BACKBONES, cfg, split


def project(features: np.ndarray, c: dict) -> np.ndarray:
    r = c["representation"]
    if r["method"] == "umap":
        import umap
        reducer = umap.UMAP(n_neighbors=r["n_neighbors"], min_dist=r["min_dist"],
                            random_state=c["seed"])
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=r["perplexity"], init="pca",
                       random_state=c["seed"])
    return reducer.fit_transform(features)


def plot_backbone(backbone: str, c=None) -> None:
    import matplotlib.pyplot as plt
    c = c or cfg()
    classes = split()["classes"]
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    for ax, iv in zip(np.ravel(axes), INTERVENTIONS):
        a, b, labels = paired_features(backbone, iv, c)
        if isinstance(b, list):   # translation: use the single documented direction
            idx = c["translation"]["directions"].index(c["representation"]["tsne_direction"])
            b = b[idx]
        emb = project(np.concatenate([a, b]), c)
        n = len(a)
        for cond, sl, marker, alpha in (("clean", slice(0, n), "o", 0.55),
                                        ("transformed", slice(n, 2 * n), "^", 0.55)):
            ax.scatter(emb[sl, 0], emb[sl, 1], c=labels, cmap="tab10", s=7,
                       marker=marker, alpha=alpha, linewidths=0, vmin=0, vmax=9)
        ax.set_title(iv, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    method = c["representation"]["method"].upper()
    fig.suptitle(f"{backbone}: {method} of clean (circle) vs transformed (triangle), colour = class",
                 fontsize=9)
    fig.tight_layout()
    savefig(fig, f"task1_{c['representation']['method']}_{backbone}", subdir="task1")


def plot_all(c=None) -> None:
    c = c or cfg()
    for bb in BACKBONES:
        plot_backbone(bb, c)
        print(f"[representation] {bb} projection saved")
