"""Task 3 Step 3 — SAM (Foret et al., 2021), standard non-adaptive version on the ERM loss.

    min_theta  max_{||eps||_2 <= rho}  L_ERM(theta + eps),   rho = 0.05.

One step on a source batch (both passes use the SAME augmented images, and the
frozen-BN policy set by the trainer — BN modules in eval mode — holds in both):
  1. L(theta), gradient g over ALL trainable parameters (backbone incl. BN
     gamma/beta, classifier head);
  2. eps = rho * g / ||g||_2 (global L2 norm); theta <- theta + eps;
  3. L(theta + eps) and its gradient;
  4. theta restored exactly (saved copy, no floating-point drift) and the base
     AdamW step (same lr / weight decay as ERM) applied with the gradient of step 3.
Weight decay is not part of the ascent gradient (AdamW decay is decoupled).
fp32 only. Logged: loss at theta (``cls_loss``), loss at theta+eps (``sam_loss``),
their difference and ||g||.
"""
import torch

from task2.methods.source_only import SourceOnly


class SAM(SourceOnly):
    name = "sam"
    uses_target = False

    def __init__(self, cfg, feat_dim, num_classes):
        super().__init__(cfg, feat_dim, num_classes)
        self.rho = float(cfg["method"]["rho"])

    def update(self, net, x, y, dom, progress, opt, scaler, amp, step):
        if amp:
            raise ValueError("SAM is implemented for fp32 only (train.amp must be false)")
        params = [p for g in opt.param_groups for p in g["params"] if p.requires_grad]
        # pass 1: ascent direction at theta
        feats, logits = net(x)
        loss, logs = self.loss(feats, logits, y, dom, progress)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in params if p.grad is not None]
        gnorm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
        scale = self.rho / (gnorm + 1e-12)
        with torch.no_grad():
            saved = [p.detach().clone() for p in params]
            for p in params:
                if p.grad is not None:
                    p.add_(p.grad, alpha=float(scale))
        # pass 2: gradient at theta + eps
        opt.zero_grad(set_to_none=True)
        feats2, logits2 = net(x)
        loss2, _ = self.loss(feats2, logits2, y, dom, progress)
        if not torch.isfinite(loss2):
            raise FloatingPointError(f"non-finite SAM loss at step {step}")
        loss2.backward()
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s)
        opt.step()
        logs.update(sam_loss=loss2.item(), sam_gap=loss2.item() - loss.item(), grad_norm=gnorm.item())
        return loss, logs
