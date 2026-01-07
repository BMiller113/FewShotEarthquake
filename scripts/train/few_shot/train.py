import os
import json
import time
from functools import partial
from typing import Any, Dict, Optional

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchnet as tnt

from protonets.engine import Engine
import protonets.utils.data as data_utils
import protonets.utils.model as model_utils
import protonets.utils.log as log_utils
from protonets.utils.checkpoint import save_checkpoint, load_checkpoint


def _now() -> float:
    return time.time()


def main(opt: Dict[str, Any]) -> None:
    # Ensure exp dir exists
    exp_dir = opt["log.exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)

    # Save opts
    opt_path = os.path.join(exp_dir, "opt.json")
    with open(opt_path, "w") as f:
        json.dump(opt, f)
        f.write("\n")

    trace_file = os.path.join(exp_dir, "trace.txt")

    # Postprocess arguments
    opt["model.x_dim"] = list(map(int, opt["model.x_dim"].split(",")))
    opt["log.fields"] = opt["log.fields"].split(",")

    # Seeds for repeatability
    torch.manual_seed(1234)
    if opt.get("data.cuda", False):
        torch.cuda.manual_seed(1234)

    # Data loaders
    if opt.get("data.trainval", False):
        data = data_utils.load(opt, ["trainval"])
        train_loader = data["trainval"]
        val_loader = None
    else:
        data = data_utils.load(opt, ["train", "val"])
        train_loader = data["train"]
        val_loader = data["val"]

    # Model
    model = model_utils.load(opt)
    if opt.get("data.cuda", False):
        model.cuda()

    engine = Engine()

    meters = {
        "train": {field: tnt.meter.AverageValueMeter() for field in opt["log.fields"]}
    }
    if val_loader is not None and not opt.get("train.no_val", False):
        meters["val"] = {field: tnt.meter.AverageValueMeter() for field in opt["log.fields"]}

    # Training control knobs
    patience = int(opt.get("train.patience", 200))
    min_delta = float(opt.get("train.min_delta", 0.0))
    val_every = int(opt.get("train.val_every", 1))
    save_every = int(opt.get("train.save_every", 1))

    # State used for early stopping / resume
    hook_state: Dict[str, Any] = {
        "best_loss": np.inf,
        "wait": 0,
        "start_epoch": 0,
    }

    def _reset_meters():
        for split, split_meters in meters.items():
            for _, meter in split_meters.items():
                meter.reset()

    # Support for resume function
    resume_path = opt.get("train.resume_path", "").strip()
    resume_payload: Optional[Dict[str, Any]] = None
    if resume_path:
        map_location = "cuda" if opt.get("data.cuda", False) else "cpu"
        resume_payload = load_checkpoint(resume_path, map_location=map_location)

        model.load_state_dict(resume_payload["model_state"])
        hook_state["best_loss"] = resume_payload.get("best_loss", np.inf)
        hook_state["wait"] = resume_payload.get("wait", 0)
        hook_state["start_epoch"] = int(resume_payload.get("epoch", 0)) + 1

        print(f"Resumed from checkpoint: {resume_path}")
        print(f"  start_epoch={hook_state['start_epoch']} best_loss={hook_state['best_loss']} wait={hook_state['wait']}")

    def on_start(state: Dict[str, Any]) -> None:
        # Clear trace for fresh runs only (not resume)
        if not resume_path and os.path.isfile(trace_file):
            os.remove(trace_file)
        state["scheduler"] = lr_scheduler.StepLR(
            state["optimizer"], step_size=int(opt["train.decay_every"]), gamma=0.5
        )

        # If resuming, restore optimizer and scheduler states (if present)
        if resume_payload is not None:
            if "optimizer_state" in resume_payload and resume_payload["optimizer_state"] is not None:
                state["optimizer"].load_state_dict(resume_payload["optimizer_state"])
            if "scheduler_state" in resume_payload and resume_payload["scheduler_state"] is not None:
                try:
                    state["scheduler"].load_state_dict(resume_payload["scheduler_state"])
                except Exception:
                    pass

            # Skip ahead epochs by setting state epoch
            state["epoch"] = hook_state["start_epoch"]

    engine.hooks["on_start"] = on_start

    def on_start_epoch(state: Dict[str, Any]) -> None:
        _reset_meters()
        state["epoch_start_time"] = _now()

    engine.hooks["on_start_epoch"] = on_start_epoch

    def on_update(state: Dict[str, Any]) -> None:
        for field, meter in meters["train"].items():
            meter.add(state["output"][field])

    engine.hooks["on_update"] = on_update

    def _should_validate(epoch_idx: int) -> bool:
        if val_loader is None:
            return False
        if opt.get("train.no_val", False):
            return False
        return (epoch_idx % max(1, val_every)) == 0

    def on_end_epoch(state: Dict[str, Any]) -> None:
        # IMPORTANT: scheduler.step() AFTER optimizer has taken steps for the epoch
        state["scheduler"].step()

        epoch_idx = int(state["epoch"])
        epoch_time = _now() - state.get("epoch_start_time", _now())

        did_val = False
        if _should_validate(epoch_idx):
            did_val = True
            model_utils.evaluate(
                state["model"],
                val_loader,
                meters["val"],
                desc=f"Epoch {epoch_idx:d} valid",
            )

        meter_vals = log_utils.extract_meter_values(meters)
        meter_vals["epoch"] = epoch_idx
        meter_vals["epoch_time_sec"] = float(epoch_time)

        # Optional: log current LR
        try:
            lr = state["optimizer"].param_groups[0]["lr"]
            meter_vals["lr"] = float(lr)
        except Exception:
            pass

        # Print progress
        print(f"Epoch {epoch_idx:04d}: {log_utils.render_meter_values(meter_vals)}")

        # Append trace
        with open(trace_file, "a") as f:
            json.dump(meter_vals, f)
            f.write("\n")

        # Save periodic "last" checkpoint for resume/restart safety
        if save_every > 0 and (epoch_idx % save_every == 0):
            save_checkpoint(
                exp_dir,
                "last.pt",
                model=state["model"],
                optimizer=state["optimizer"],
                scheduler=state["scheduler"],
                epoch=epoch_idx,
                best_loss=float(hook_state["best_loss"]) if np.isfinite(hook_state["best_loss"]) else None,
                wait=int(hook_state["wait"]),
            )

        # Early stopping + best model
        if did_val:
            val_loss = float(meter_vals["val"]["loss"])

            # Improvement test with min_delta
            improved = val_loss < (hook_state["best_loss"] - min_delta)

            if improved:
                hook_state["best_loss"] = val_loss
                hook_state["wait"] = 0
                print(f"==> best model (val loss = {val_loss:0.6f}), saving model...")

                # Save full model object for compatibility with existing eval.py
                state["model"].cpu()
                torch.save(state["model"], os.path.join(exp_dir, "best_model.pt"))
                if opt.get("data.cuda", False):
                    state["model"].cuda()

                # Save resumable checkpoint snapshot
                save_checkpoint(
                    exp_dir,
                    "best_state.pt",
                    model=state["model"],
                    optimizer=state["optimizer"],
                    scheduler=state["scheduler"],
                    epoch=epoch_idx,
                    best_loss=float(hook_state["best_loss"]),
                    wait=int(hook_state["wait"]),
                )
            else:
                hook_state["wait"] += 1
                if hook_state["wait"] > patience:
                    print(f"==> patience {patience:d} exceeded (min_delta={min_delta:g}); stopping.")
                    state["stop"] = True
        else:
            # No validation mode: still write a best_model at end of epoch for convenience
            state["model"].cpu()
            torch.save(state["model"], os.path.join(exp_dir, "best_model.pt"))
            if opt.get("data.cuda", False):
                state["model"].cuda()

    engine.hooks["on_end_epoch"] = partial(on_end_epoch)

    # --- Run training (including ctrl-c safety)
    try:
        engine.train(
            model=model,
            loader=train_loader,
            optim_method=getattr(optim, opt["train.optim_method"]),
            optim_config={"lr": opt["train.learning_rate"], "weight_decay": opt["train.weight_decay"]},
            max_epoch=opt["train.epochs"],
        )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Saving last checkpoint before exit...")

        # best effort save
        try:
            save_checkpoint(
                exp_dir,
                "last.pt",
                model=model,
                optimizer=None,
                scheduler=None,
                epoch=int(getattr(engine, "epoch", 0)),
                best_loss=float(hook_state["best_loss"]) if np.isfinite(hook_state["best_loss"]) else None,
                wait=int(hook_state["wait"]),
            )
        except Exception as e:
            print(f"Failed to save last checkpoint on interrupt: {e}")

        raise
