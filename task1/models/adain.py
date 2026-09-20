"""AdaIN style transfer for cue-conflict generation.

The encoder/decoder architecture and the AdaIN operation below are taken from
the public implementation naoto0804/pytorch-AdaIN (MIT licence), which is the
reference PyTorch implementation of Huang & Belongie (2017). Pretrained weights
(vgg_normalised.pth, decoder.pth) are downloaded from that repository's release.
Attribution is recorded in the top-level README.

Usage: content and style images are float tensors in [0,1] (no ImageNet
normalisation: the VGG's first 1x1 convolution performs the normalisation).
"""
import urllib.request

import torch
import torch.nn as nn

from task1.configs import ckpt_dir

WEIGHT_URLS = {
    "vgg_normalised.pth": "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth",
    "decoder.pth": "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth",
}


def calc_mean_std(feat, eps: float = 1e-5):
    N, C = feat.size()[:2]
    std = (feat.view(N, C, -1).var(dim=2) + eps).sqrt().view(N, C, 1, 1)
    mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def adaptive_instance_normalization(content_feat, style_feat):
    """Replace the per-channel mean/std of the content features with the style's."""
    size = content_feat.size()
    s_mean, s_std = calc_mean_std(style_feat)
    c_mean, c_std = calc_mean_std(content_feat)
    normalized = (content_feat - c_mean.expand(size)) / c_std.expand(size)
    return normalized * s_std.expand(size) + s_mean.expand(size)


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


def adain_weights_dir():
    d = ckpt_dir() / "adain"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_weights(verbose: bool = True) -> None:
    d = adain_weights_dir()
    for name, url in WEIGHT_URLS.items():
        p = d / name
        if p.exists() and p.stat().st_size > 1_000_000:
            if verbose:
                print(f"[adain] {name} already present ({p.stat().st_size/1e6:.0f} MB)")
            continue
        if verbose:
            print(f"[adain] downloading {name} ...")
        tmp = p.with_suffix(".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(p)
        if verbose:
            print(f"[adain] saved {name} ({p.stat().st_size/1e6:.0f} MB)")


class AdaINStyleTransfer(nn.Module):
    """Encoder (VGG up to relu4_1) -> AdaIN -> decoder."""

    def __init__(self, device="cuda"):
        super().__init__()
        download_weights()
        d = adain_weights_dir()
        v = vgg
        v.load_state_dict(torch.load(d / "vgg_normalised.pth", map_location="cpu", weights_only=False))
        self.encoder = nn.Sequential(*list(v.children())[:31])   # up to relu4_1
        self.decoder = decoder
        self.decoder.load_state_dict(torch.load(d / "decoder.pth", map_location="cpu", weights_only=False))
        self.eval().to(device)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, content, style, alpha: float = 1.0):
        """content/style: float [N,3,H,W] in [0,1]; alpha interpolates between the
        content features (0) and the fully stylised features (1)."""
        f_c, f_s = self.encoder(content), self.encoder(style)
        t = adaptive_instance_normalization(f_c, f_s)
        t = alpha * t + (1 - alpha) * f_c
        return self.decoder(t).clamp(0, 1)
