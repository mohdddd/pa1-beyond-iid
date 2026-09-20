"""Frozen backbone wrappers (ResNet-50, ViT-B/16, CLIP ViT-B-32) + feature cache.

Each wrapper takes the common base images (uint8 [N,3,224,224]), applies ITS OWN
normalisation, and returns the final representation:
  resnet50    -> 2048-d global-average-pooled feature (fc -> Identity)
  vit_b16     -> 768-d final class token (after final LayerNorm; heads -> Identity)
  clip_vitb32 -> 512-d L2-normalised CLIP image embedding
Features are cached on Drive as features/<backbone>/<condition>.npy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

from task1.configs import feature_dir

FEATURE_DIM = {"resnet50": 2048, "vit_b16": 768, "clip_vitb32": 512}


class FrozenBackbone(nn.Module):
    def __init__(self, name, net, mean, std, is_clip=False, tokenizer=None):
        super().__init__()
        self.name, self.net, self.is_clip, self.tokenizer = name, net, is_clip, tokenizer
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.net.eval()

    def train(self, mode: bool = True):   # always frozen / eval mode
        return super().train(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0 if x.dtype == torch.uint8 else x.float()
        x = (x - self.mean) / self.std
        if self.is_clip:
            return F.normalize(self.net.encode_image(x).float(), dim=-1)
        return self.net(x).float()

    @torch.no_grad()
    def clip_text_classifier(self, classnames, template: str):
        """Zero-shot weights: normalised text embeddings [C,512] and CLIP's logit scale."""
        assert self.is_clip
        tokens = self.tokenizer([template.format(c) for c in classnames]).to(self.mean.device)
        w = F.normalize(self.net.encode_text(tokens).float(), dim=-1)
        return w, float(self.net.logit_scale.exp().item())


def load_backbone(name: str, device="cuda") -> FrozenBackbone:
    if name == "resnet50":
        w = tvm.ResNet50_Weights.IMAGENET1K_V2
        net = tvm.resnet50(weights=w)
        net.fc = nn.Identity()
        t = w.transforms()
        bb = FrozenBackbone(name, net, t.mean, t.std)
    elif name == "vit_b16":
        w = tvm.ViT_B_16_Weights.IMAGENET1K_V1
        net = tvm.vit_b_16(weights=w)
        net.heads = nn.Identity()
        t = w.transforms()
        bb = FrozenBackbone(name, net, t.mean, t.std)
    elif name == "clip_vitb32":
        import open_clip
        from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
        net, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        bb = FrozenBackbone(name, net, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD, is_clip=True,
                            tokenizer=open_clip.get_tokenizer("ViT-B-32"))
    else:
        raise ValueError(name)
    return bb.to(device)


# ---------- feature extraction + cache ----------
@torch.no_grad()
def extract(backbone, images: torch.Tensor, batch_size: int, device) -> np.ndarray:
    """uint8 [N,3,224,224] -> float32 [N,D] (fp32 inference)."""
    out = [backbone(images[i:i + batch_size].to(device)).cpu() for i in range(0, len(images), batch_size)]
    return torch.cat(out).numpy()


def feature_path(backbone: str, condition: str):
    return feature_dir(backbone) / f"{condition}.npy"


def has_features(backbone: str, condition: str) -> bool:
    return feature_path(backbone, condition).exists()


def load_features(backbone: str, condition: str) -> np.ndarray:
    return np.load(feature_path(backbone, condition))


def save_features(backbone: str, condition: str, feats: np.ndarray) -> None:
    np.save(feature_path(backbone, condition), feats.astype(np.float32))
