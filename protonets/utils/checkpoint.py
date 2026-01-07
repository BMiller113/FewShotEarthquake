import os
from typing import Any, Dict, Optional

import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_checkpoint(
    exp_dir: str,
    filename: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int,
    best_loss: Optional[float] = None,
    wait: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Saves a training checkpoint as a dict (NOT a full model object).
    This is intended for resuming training robustly.
    """
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "best_loss": best_loss,
        "wait": wait,
    }

    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        try:
            payload["scheduler_state"] = scheduler.state_dict()
        except Exception:
            # Some schedulers may not have state_dict; ignore safely.
            payload["scheduler_state"] = None

    if extra is not None:
        payload["extra"] = extra

    out_path = os.path.join(ckpt_dir, filename)
    torch.save(payload, out_path)
    return out_path


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Loads a checkpoint dict produced by save_checkpoint().
    """
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
