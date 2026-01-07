import os
import json
import shutil
import subprocess
import sys

from protonets.utils import format_opts, merge_dict
from protonets.utils.log import load_trace


def main(opt):
    # Repo root (CWD-independent)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    # Canonical results dir (you confirmed this is where training writes)
    results_dir = os.path.join(repo_root, "results")
    trainval_dir = os.path.join(results_dir, "trainval")
    os.makedirs(trainval_dir, exist_ok=True)

    # Read trace/opts from the base training run
    trace_file = os.path.join(results_dir, "trace.txt")
    opt_file = os.path.join(results_dir, "opt.json")

    if not os.path.exists(trace_file):
        raise FileNotFoundError(f"Missing training trace: {trace_file}")
    if not os.path.exists(opt_file):
        raise FileNotFoundError(f"Missing training options: {opt_file}")

    trace_vals = load_trace(trace_file)
    best_epoch = trace_vals["val"]["loss"].argmin()

    with open(opt_file, "r") as f:
        model_opt = json.load(f)

    # Override options for trainval rerun
    model_opt = merge_dict(
        model_opt,
        {
            "log.exp_dir": trainval_dir,
            "data.trainval": True,
            "train.epochs": int(best_epoch) + int(model_opt.get("train.patience", 0)),
        },
    )
    run_train_py = os.path.join(repo_root, "scripts", "train", "few_shot", "run_train.py")
    py = sys.executable

    subprocess.check_call([py, run_train_py] + format_opts(model_opt), cwd=repo_root)

    # Ensure eval has opt.json next to the trainval model (eval.py expects that)
    shutil.copy2(opt_file, os.path.join(trainval_dir, "opt.json"))
    shutil.copy2(trace_file, os.path.join(trainval_dir, "trace.txt"))
    src_model = os.path.join(results_dir, "best_model.pt")
    dst_model = os.path.join(trainval_dir, "best_model.pt")
    if os.path.exists(src_model) and not os.path.exists(dst_model):
        shutil.copy2(src_model, dst_model)

    if not os.path.exists(dst_model):
        raise FileNotFoundError(
            "Trainval finished but no best_model.pt found in results/trainval.\n"
            f"Expected: {dst_model}\n"
            f"Also checked: {src_model}\n"
        )
