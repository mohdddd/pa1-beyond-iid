"""Linear classifier heads on frozen features, and the CLIP zero-shot classifier."""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.logging_utils import CSVLogger
from common.seed import make_generator, set_seed
from task1.configs import ckpt_dir


def train_linear_head(Xtr, ytr, Xva, yva, num_classes, hcfg, seed, device, log_path):
    """AdamW + early stopping on validation accuracy; returns (best head, summary)."""
    set_seed(seed)   # identical init + shuffling seed for every backbone's head
    Xtr, Xva = torch.as_tensor(Xtr, device=device), torch.as_tensor(Xva, device=device)
    ytr = torch.as_tensor(ytr, device=device, dtype=torch.long)
    yva = torch.as_tensor(yva, device=device, dtype=torch.long)

    head = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=hcfg["lr"], weight_decay=hcfg["weight_decay"])
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=hcfg["batch_size"],
                        shuffle=True, generator=make_generator(seed))
    lossf = nn.CrossEntropyLoss()
    log = CSVLogger(log_path, resume=False)

    best_acc, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(1, hcfg["max_epochs"] + 1):
        head.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            loss = lossf(head(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        head.eval()
        with torch.no_grad():
            val_loss = lossf(head(Xva), yva).item()
            tr_acc = (head(Xtr).argmax(1) == ytr).float().mean().item()
            va_acc = (head(Xva).argmax(1) == yva).float().mean().item()
        log.log({"epoch": epoch, "train_loss": tot / n, "train_acc": tr_acc,
                 "val_loss": val_loss, "val_acc": va_acc})
        if va_acc > best_acc:
            best_acc, best_state, best_epoch, bad = va_acc, copy.deepcopy(head.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= hcfg["patience"]:
                break
    head.load_state_dict(best_state)
    return head, {"best_epoch": best_epoch, "best_val_acc": best_acc, "epochs_run": epoch}


def save_head(head, model_name: str) -> None:
    torch.save(head.state_dict(), ckpt_dir() / f"head_{model_name}.pt")


def load_head(model_name: str, device):
    state = torch.load(ckpt_dir() / f"head_{model_name}.pt", map_location=device, weights_only=False)
    head = nn.Linear(state["weight"].shape[1], state["weight"].shape[0]).to(device)
    head.load_state_dict(state)
    return head.eval()


ZEROSHOT_FILE = "clip_zeroshot_text.pt"


def save_zeroshot(weights, logit_scale, prompt, classes) -> None:
    torch.save({"weights": weights.cpu(), "logit_scale": logit_scale, "prompt": prompt,
                "classes": classes}, ckpt_dir() / ZEROSHOT_FILE)


def has_zeroshot() -> bool:
    return (ckpt_dir() / ZEROSHOT_FILE).exists()


def load_zeroshot(device):
    z = torch.load(ckpt_dir() / ZEROSHOT_FILE, map_location=device, weights_only=False)
    return z["weights"].to(device), z["logit_scale"]
