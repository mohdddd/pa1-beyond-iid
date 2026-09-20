"""Controlled interventions on the common 224x224 base images.

Every function takes and returns uint8 [N,3,224,224] tensors, so each condition
is stored once and reused by all three backbones.

colour     : grayscale (removes colour, keeps luminance + geometry)
             hue_rotation / palette_transfer (changes colour, keeps geometry)
translation: reflection padding + shifted crop (object displaced, content kept)
patch      : 4x4 grid permutation (local evidence kept, global layout destroyed)
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

DIRECTION_OFFSETS = {  # (row_start, col_start) into the padded image, pad = delta
    "right": (1, 0), "left": (1, 2), "down": (0, 1), "up": (2, 1),
}


def _to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


# ---------------- colour ----------------
def grayscale(images: torch.Tensor) -> torch.Tensor:
    """Removes chromatic information; luminance, edges and geometry unchanged."""
    return _to_uint8(TF.rgb_to_grayscale(_to_float(images), num_output_channels=3))


def hue_rotation(images: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotates every hue by a fixed angle in HSV. Saturation, value (hence
    luminance ordering, edges and geometry) are preserved; only the colour
    identity of each surface changes."""
    factor = (degrees / 360.0)
    factor = (factor + 0.5) % 1.0 - 0.5          # adjust_hue expects [-0.5, 0.5]
    return _to_uint8(TF.adjust_hue(_to_float(images), factor))


def palette_transfer(images: torch.Tensor, labels, seed: int):
    """Per-channel mean/std matching to a donor image of a DIFFERENT class:
    global colour statistics are swapped, object geometry is preserved.
    Returns (images, donor_indices)."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    x = _to_float(images)
    mu = x.mean(dim=(2, 3), keepdim=True)
    sd = x.std(dim=(2, 3), keepdim=True).clamp_min(1e-5)
    donors = []
    for i in range(len(x)):
        pool = np.where(labels != labels[i])[0]
        donors.append(int(rng.choice(pool)))
    d = torch.as_tensor(donors)
    out = (x - mu) / sd * sd[d] + mu[d]
    return _to_uint8(out), donors


# ---------------- translation ----------------
def translate(images: torch.Tensor, delta: int, direction: str) -> torch.Tensor:
    """Reflection padding followed by a shifted crop (image content moves by
    `delta` pixels in `direction`; output stays 224x224)."""
    if delta == 0:
        return images.clone()
    x = _to_float(images)
    p = delta
    x = F.pad(x, (p, p, p, p), mode="reflect")
    r, c = DIRECTION_OFFSETS[direction]
    h, w = images.shape[-2:]
    return _to_uint8(x[:, :, r * p:r * p + h, c * p:c * p + w])


# ---------------- patch structure ----------------
def _split_patches(x: torch.Tensor, g: int) -> torch.Tensor:
    n, c, h, w = x.shape
    ph, pw = h // g, w // g
    return (x.reshape(n, c, g, ph, g, pw).permute(0, 2, 4, 1, 3, 5).reshape(n, g * g, c, ph, pw))


def _join_patches(p: torch.Tensor, g: int) -> torch.Tensor:
    n, _, c, ph, pw = p.shape
    return (p.reshape(n, g, g, c, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(n, c, g * ph, g * pw))


def patch_shuffle(images: torch.Tensor, grid: int, seed: int):
    """One non-identity permutation of the grid x grid pixel-space patches per
    image (seeded, so every model sees identical shuffled images).
    Returns (images, permutations)."""
    rng = np.random.default_rng(seed)
    k = grid * grid
    patches = _split_patches(images, grid)
    perms = []
    for i in range(len(images)):
        perm = rng.permutation(k)
        while np.array_equal(perm, np.arange(k)):     # reject the identity
            perm = rng.permutation(k)
        patches[i] = patches[i][torch.as_tensor(perm)]
        perms.append(perm.tolist())
    return _join_patches(patches, grid), perms
